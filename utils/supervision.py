import torch
from torch import nn

MAX_FLOW = 400

def segmentation_edges(segmentation, blur_size = 9):
    # Blur size must be odd
    blur_size = blur_size if blur_size % 2 else blur_size + 1
    blur_kernel_size = (blur_size, blur_size)
    blur_pad = (blur_size//2, blur_size//2)

    # A Laplacian kernel to compute gradients
    lap_kernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap_kernel = lap_kernel.expand(1, 1, 3, 3)

    edges = nn.functional.conv2d(segmentation[:,None,:,:], lap_kernel.cuda(), stride=1, padding=1)
    edges[edges <= 0] = 0
    edges[edges > 0] = 1

    # Blur kernel for smoother edges
    if not blur_size <= 0:
        blur_kernel = torch.ones(blur_kernel_size).expand(1, 1, *blur_kernel_size)
        blur_kernel = blur_kernel / blur_kernel.sum()

        edges = nn.functional.conv2d(edges, blur_kernel.cuda(), stride=1, padding=blur_pad)

    return edges


import torch.nn.functional as F

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
from .label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from torchvision import transforms
from functools import partial
from operator import itemgetter

# Tools
def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

# Active Boundary Loss
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_lr = gt[:,:,1:]-gt[:,:,:-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr = torch.nn.functional.pad(gt_lr, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).cuda() # 8k
        kl_maps = torch.zeros((0,len(x))).cuda() # 8k

        kl_center = logits[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, logits, target):
        eps = 1e-10
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt) # careful
        
        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss


def sequence_loss(flow_preds, flow_gt, valid, seg_out, seg_gt, segloss_fn, gamma=0.8, lambda_ = 0.5, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (valid >= 0.5) & (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    """ Segmentation Loss """

    # Experimental edges loss
    # Old:
    # gt_edges = segmentation_edges(seg_gt)
    # seg_pred = seg_out.max(dim=1)[1].float()
    # pred_edges = segmentation_edges(seg_pred, blur_size=0)
    # edges_loss = (gt_edges - pred_edges).abs().mean()

    # New:
    # abl = ABL()
    # abl_loss = abl(seg_out, seg_gt)

    seg_gt[valid == 0] = 255
    seg_loss = segloss_fn(seg_out, seg_gt.long())

    total_loss = flow_loss + lambda_ * seg_loss

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'seg_loss':seg_loss
    }

    return total_loss, metrics