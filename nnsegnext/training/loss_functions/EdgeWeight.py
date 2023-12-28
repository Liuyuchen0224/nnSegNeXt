from torch import nn
import SimpleITK as sitk
import torch
from nnsegnext.utilities.to_torch import maybe_to_torch, to_cuda
from nnsegnext.utilities.tensor_utilities import sum_tensor

def CannyEdgeDetectionImage(image,LH,UH,Var):
    canny_op = sitk.CannyEdgeDetectionImageFilter()
    canny_op.SetLowerThreshold(LH)
    canny_op.SetUpperThreshold(UH)
    canny_op.SetVariance(Var)
    canny_op.SetMaximumError(0.5)
    image_canny = canny_op.Execute(image)
    image_canny = sitk.Cast(image_canny, sitk.sitkInt16)
    return sitk.GetArrayFromImage(image_canny)


    
def EdgeWeight(data, target):
    batchsize = data.size()[0]
    weights = []
    for i in range(batchsize):    
        imgdata = sitk.GetImageFromArray(data[i,0,:,:,:])
        imgtarget = sitk.GetImageFromArray(target[0][i,0,:,:,:])
        dataedge = CannyEdgeDetectionImage(imgdata,0.1,1,1)
        targetedge = CannyEdgeDetectionImage(imgtarget,0.1,1,1)
        smooth = 1e-6

        dataedge = maybe_to_torch(dataedge)
        targetedge = maybe_to_torch(targetedge)

        if torch.cuda.is_available():
            dataedge = to_cuda(dataedge)
            targetedge = to_cuda(targetedge)

        axes = list(range(len(dataedge.shape)))
        tp = dataedge * targetedge
        fp = dataedge * (1 - targetedge)
        fn = (1 - dataedge) * targetedge
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        dc = ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).mean()
        # dc = 1 - dc

        # weight = 0.5 + dc       
        weights.append(dc)
    return torch.tensor(weights)