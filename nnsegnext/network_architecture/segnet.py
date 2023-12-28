import torch
import numpy as np
from nnsegnext.network_architecture.neural_network import SegmentationNetwork
import torch.nn as nn
import torch.nn.functional as F

class SegNet(SegmentationNetwork):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv3d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm3d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm3d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv64 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn64 = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm3d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm3d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv64d = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn64d = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv3d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm3d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm3d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm3d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm3d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv3d(64, label_nbr, kernel_size=3, padding=1)

        self.num_classes=4
        self._deep_supervision = False
        self.do_ds = False

    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool3d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool3d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x64 = F.relu(self.bn64(self.conv64(x31)))
        x33 = F.relu(self.bn33(self.conv33(x64)))
        x3p, id3 = F.max_pool3d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool3d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool3d(x53,kernel_size=2, stride=2,return_indices=True)


        # Stage 5d
        x5d = F.max_unpool3d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool3d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool3d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x64d = F.relu(self.bn64d(self.conv64d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x64d)))

        # Stage 2d
        x2d = F.max_unpool3d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool3d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
