import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy.metric import binary

import GeodisTK
from scipy import ndimage

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
        
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim==2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim==len(g.shape))
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s)
    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def hd(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return  hd95
    else:
        return 0
        
def process_label(label):
    csf = np.zeros_like(label)
    gm = np.zeros_like(label)
    wm = np.zeros_like(label)
    csf [label == 1] = 1
    gm [label == 2] = 1
    wm [label == 3] = 1
    return csf,gm,wm

def test():
    hcppath='/home/liuyc/PaperProject/Datasets/TVT_dataset/Task001_BrainTissueHCPTVT/test/label'
    ixipath='/home/liuyc/PaperProject/Datasets/TVT_dataset/Task002_BrainTissueIXITVT/test/label'    
    saldpath='/home/liuyc/PaperProject/Datasets/TVT_dataset/Task003_BrainTissueSALDTVT/test/label'    
    ibsrpath='/home/liuyc/PaperProject/Datasets/TVT_dataset/Task004_BrainTissueIBSRTVT/test/label'        
    path = '/home/liuyc/PaperProject/Datasets/infers/'
    filelist  = os.listdir(path)
    dir='Segnet'
    filelist  = os.listdir(path+dir)
    for  file in  filelist:
        if os.path.isdir(f'{path+dir}/{file}'):
            if file == 'Segnet_HCP_IXI':
                    network = file.split('_')[0]
                    PretrainDataset = file.split('_')[1]
                    InferDataset = file.split('_')[2]
                    if  InferDataset == 'HCP':
                        dataset  = hcppath
                    elif  InferDataset == 'IXI':
                        dataset  = ixipath
                    elif  InferDataset == 'SALD':
                        dataset  = saldpath
                    elif  InferDataset == 'IBSR':
                        dataset  = ibsrpath
                    else:
                        raise ValueError(-1)
                # if not os.path.exists(f'{os.path.join(path,file)}/dice_pre.txt'):
                    print(network,PretrainDataset,InferDataset)
                    label_list=sorted(glob.glob(os.path.join(dataset,'*nii.gz')))
                    infer_list=sorted(glob.glob(os.path.join(path+dir,file,'*nii.gz')))
                    if  len(label_list)!= len(infer_list):
                        print("WRONG SIZE")
                    print("loading success...")
                    Dice_csf=[]
                    Dice_gm=[]
                    Dice_wm=[]

                    HD_csf=[]
                    HD_gm=[] 
                    HD_wm=[]

                    fw = open(f'{os.path.join(path+dir,file)}/dice_pre.txt', 'w')

                    for label_path,infer_path in zip(label_list,infer_list):
                        print(label_path.split('/')[-1])
                        print(infer_path.split('/')[-1])
                        label,infer = read_nii(label_path),read_nii(infer_path)
                        label_csf,label_gm,label_wm=process_label(label)
                        infer_csf,infer_gm,infer_wm=process_label(infer)
                        Dice_csf.append(dice(infer_csf,label_csf))
                        Dice_gm.append(dice(infer_gm,label_gm))
                        Dice_wm.append(dice(infer_wm,label_wm))

                        HD_csf.append(binary_hausdorff95(infer_csf,label_csf))
                        HD_gm.append(binary_hausdorff95(infer_gm,label_gm))
                        HD_wm.append(binary_hausdorff95(infer_wm,label_wm))

                        fw.write('*'*20+'\n',)
                        fw.write(infer_path.split('/')[-1]+'\n')
                        fw.write('hd_csf: {:.4f}\n'.format(HD_csf[-1]))
                        fw.write('hd_gm: {:.4f}\n'.format(HD_gm[-1]))
                        fw.write('hd_wm: {:.4f}\n'.format(HD_wm[-1]))
                        fw.write('*'*20+'\n',)
                        fw.write('Dice_csf: {:.4f}\n'.format(Dice_csf[-1]))
                        fw.write('Dice_gm: {:.4f}\n'.format(Dice_gm[-1]))
                        fw.write('Dice_wm: {:.4f}\n'.format(Dice_wm[-1]))

                        #print('dice_csf: {:.4f}'.format(np.mean(Dice_csf)))
                        #print('dice_gm: {:.4f}'.format(np.mean(Dice_gm)))
                        #print('dice_wm: {:.4f}'.format(np.mean(Dice_wm)))
                    dsc = [np.mean(Dice_csf), np.mean(Dice_gm), np.mean(Dice_wm)]
                    avg_hd = [np.mean(HD_csf), np.mean(HD_gm), np.mean(HD_wm)]
                    fw.write(f'Dice_csf{str(np.mean(Dice_csf))} ' + '\n')
                    fw.write(f'Dice_gm{str(np.mean(Dice_gm))} ' + '\n')
                    fw.write(f'Dice_wm{str(np.mean(Dice_wm))} ' + '\n')

                    fw.write(f'HD_csf{str(np.mean(HD_csf))} ' + '\n')
                    fw.write(f'HD_gm{str(np.mean(HD_gm))} ' + '\n')
                    fw.write(f'HD_wm{str(np.mean(HD_wm))} ' + '\n')

                    fw.write(f'Dice{str(np.mean(dsc))} ' + '\n')
                    fw.write(f'HD{str(np.mean(avg_hd))} ' + '\n')
                    fw.write('************************' + '\n')
                    
                    fw.write('%.2f' % (np.mean(Dice_csf)*100)+'&'+
                            '%.2f' % (np.mean(HD_csf))+'&'+
                            '%.2f' % (np.mean(Dice_gm)*100)+'&'+
                            '%.2f' % (np.mean(HD_gm))+'&'+
                            '%.2f' % (np.mean(Dice_wm)*100)+'&'+
                            '%.2f' % (np.mean(HD_wm))+'&'+
                            '%.2f' % (np.mean(dsc)*100)+'&'+
                            '%.2f' % (np.mean(avg_hd))                                                    
                                    )
                    #print('Dice'+str(np.mean(dsc))+' '+'\n')
                    #print('HD'+str(np.mean(avg_hd))+' '+'\n')
        


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-network",default='nnunet_hcp', help="fold name")
    # args = parser.parse_args()
    # network=args.network
    test()
