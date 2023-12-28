import torch
from torch import nn
from nnsegnext.network_architecture.neural_network import SegmentationNetwork
import numpy as np
from copy import deepcopy
from timm.models.layers import DropPath

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self,input_features,output_features,conv_kwargs,dropout_op_kwargs,norm_op_kwargs,nonlin_kwargs) -> None:
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
            
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        
        
        if self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = nn.Dropout3d(**self.dropout_op_kwargs)
        else:
            self.dropout = None
                        
        self.conv = nn.Conv3d(input_features,output_features, **conv_kwargs)
        self.instnorm = nn.InstanceNorm3d(output_features, **self.norm_op_kwargs)
        self.lrelu = nn.LeakyReLU(**self.nonlin_kwargs)
               
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))  
    
class ConvLayers(nn.Module):
    def __init__(self,input_feature_channels,output_feature_channels,num_convs,first_stride,conv_kwargs,dropout_op_kwargs,norm_op_kwargs,nonlin_kwargs) -> None:
        super().__init__()
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}


        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        
        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs
        
        self.blocks = nn.Sequential(
            *([ConvDropoutNormNonlin(self.input_channels, self.output_channels,
                           self.conv_kwargs_first_conv,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs)]+
              [ConvDropoutNormNonlin(self.output_channels, self.output_channels,
                           self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs) for _ in range(num_convs - 1)]))
    
    def forward(self, x):
        return self.blocks(x)
    
class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (1, 1, 7), padding=(0, 0, 3), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim)
        self.conv0_3 = nn.Conv3d(dim, dim, (1, 7, 1), padding=(0, 3, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (1, 1, 11), padding=(0, 0, 5), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (11, 1, 1), padding=(5, 0, 0), groups=dim)
        self.conv1_3 = nn.Conv3d(dim, dim, (1, 11, 1), padding=(0, 5, 0), groups=dim)
        
        self.conv2_1 = nn.Conv3d(
            dim, dim, (1, 1, 21), padding=(0, 0, 10), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (21, 1, 1), padding=(10, 0, 0), groups=dim)
        self.conv2_3 = nn.Conv3d(
            dim, dim, (1, 21, 1), padding=(0, 10, 0), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.conv0_3(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.conv1_3(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.conv2_3(attn_2)
        
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class Block(nn.Module):

    def __init__(self,
                 dim,
                 out_features,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                    ):
        super().__init__()
        self.input_channels = dim
        self.output_channels = out_features
        self.norm1 = nn.InstanceNorm3d(self.input_channels)
        self.attn = SpatialAttention(self.input_channels)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.InstanceNorm3d(self.input_channels)
        mlp_hidden_dim = int(self.input_channels * mlp_ratio)
        self.mlp = Mlp(in_features=self.input_channels, hidden_features=mlp_hidden_dim,out_features=self.output_channels,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((self.input_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((self.input_channels)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        return x    
        
    
class Segnext(SegmentationNetwork):
    MAX_NUM_3D = 320
    def __init__(self,input_channels,num_classes,dims,conv_kwargs,softmax_helper,dropout_op_kwargs,norm_op_kwargs,nonlin_kwargs,deep_supervision,weightInitializer) -> None:
        super().__init__()  

        if nonlin_kwargs is None:
            self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            self.dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            self.norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            self.conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        
        self.num_classes = num_classes
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.tu = []
        self.seg_outputs = []
        self.weightInitializer = weightInitializer
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.final_nonlin = softmax_helper
        self.drop_path_rate=0.1
        num_pool = len(dims)
        output_features = dims
        input_features = input_channels
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                num_pool)]  # stochastic depth decay rule
        self.pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        
        for d in range(num_pool):
            # add convolutions
            first_stride = self.pool_op_kernel_sizes[d-1] if d != 0 else None
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            output_features = dims[d]
            
            if d>=2:
                self.conv_blocks_context.append(nn.Sequential(
                    ConvLayers(input_features, output_features,1,first_stride,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs),
                    Block(output_features,output_features,mlp_ratio=4.,drop=0.,drop_path=dpr[d],act_layer=nn.GELU)))
            else:
                self.conv_blocks_context.append(nn.Sequential(
                    ConvLayers(input_features, output_features,1,first_stride,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs),
                    ConvLayers(output_features, output_features,1,None,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs)))
                
            input_features = output_features
            output_features = int(np.round(output_features*2))
            output_features = min(output_features, self.MAX_NUM_3D)
        first_stride = self.pool_op_kernel_sizes[-1]    
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        final_num_features = self.conv_blocks_context[-1][1].output_channels 
        self.conv_blocks_context.append(nn.Sequential(
            ConvLayers(input_features, output_features,1,first_stride,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs),
            Block(output_features,final_num_features,mlp_ratio=4.,drop=0.,drop_path=dpr[d],act_layer=nn.GELU)))
        
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)][1].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2 

            if u != num_pool - 1:
                final_num_features = self.conv_blocks_context[-(3 + u)][1].output_channels
            else:
                final_num_features = nfeatures_from_skip 

            final_num_features = nfeatures_from_skip

            self.tu.append(nn.ConvTranspose3d(nfeatures_from_down,nfeatures_from_skip,self.pool_op_kernel_sizes[-(u + 1)],
                                          self.pool_op_kernel_sizes[-(u + 1)], bias=False)) 
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]    
            if u <= 2:
                self.conv_blocks_localization.append(nn.Sequential(
                    ConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip,1,None,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs),
                    Block(nfeatures_from_skip,final_num_features,mlp_ratio=4.,drop=0.,drop_path=dpr[d],act_layer=nn.GELU)))
            else:
                self.conv_blocks_localization.append(nn.Sequential(
                    ConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip,1,None,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs),
                    ConvLayers(nfeatures_from_skip, final_num_features,1,None,self.conv_kwargs,self.dropout_op_kwargs,self.norm_op_kwargs,self.nonlin_kwargs)))

        self.seg_outputs.extend(nn.Conv3d(item[-1].output_channels, self.num_classes, 1, 1, 0, 1, 1, False) for item in self.conv_blocks_localization)

        self.upscale_logits_ops = [lambda x: x for _ in range(num_pool - 1)]
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
    
    def forward(self, x):
        #print(x.shape)
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            
            skips.append(x)
            
        x = self.conv_blocks_context[-1](x)
        
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            
            x = self.conv_blocks_localization[u](x)
           
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            
        if self._deep_supervision and self.do_ds:
            
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]   