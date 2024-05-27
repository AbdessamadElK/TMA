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
    # gt_edges = segmentation_edges(seg_gt)
    # seg_pred = seg_out.max(dim=1)[1].float()
    # pred_edges = segmentation_edges(seg_pred, blur_size=0)
    # edges_loss = (gt_edges - pred_edges).abs().mean()

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