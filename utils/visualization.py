
import numpy
from matplotlib import colors
from skimage import io
import cv2

def visualize_optical_flow(flow, savepath=None, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    flow = flow.transpose(1,2,0)
    flow[numpy.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = numpy.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = numpy.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = numpy.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=numpy.pi*2
    hsv[..., 0] = ang/numpy.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)
    # This all seems like an overkill, but it's just to exactly match the cv2 implementation
    bgr = numpy.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)


    if savepath is not None:
        out = bgr*255
        io.imsave(savepath, out.astype('uint8'))

    return bgr, (mag.min(), mag.max())


def writer_add_features(tensor_feat):
    feat_img = tensor_feat.detach().cpu().numpy()
    # img_grid = self.make_grid(feat_img)
    feat_img = numpy.sum(feat_img,axis=0)
    feat_img = feat_img -numpy.min(feat_img)
    img_grid = 255*feat_img/numpy.max(feat_img)
    img_grid = cv2.applyColorMap(numpy.array(img_grid, dtype=numpy.uint8), cv2.COLORMAP_JET)
    return img_grid


import numpy as np

from .cityscapes_labels import labels_19

def segmentation2rgb_19(seg:np.ndarray)-> np.ndarray:
    # Visualize semantic segmentation according to cityscapes labels
    # Seg.shape : [h, w]
    h, w = seg.shape
    visualization = np.zeros((h, w, 3))

    for label in filter(lambda label : 0 <= label.trainId < 19, labels_19):
        visualization[seg == label.trainId] = np.array(label.color)

    return visualization.astype('uint8')