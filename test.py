import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
from pathlib import Path
import imageio
from tqdm import tqdm

from dataloader.dsec_full import DSECfull
from model.TMA import TMA

import flow_vis
from utils.visualization import visualize_optical_flow, segmentation2rgb_19


def save_flow_submission(save_dir: Path, flow: np.ndarray, file_index: int):
    # flow_u(u,v) = ((float)I(u,v,1)-2^15)/128.0;
    # flow_v(u,v) = ((float)I(u,v,2)-2^15)/128.0;
    # valid(u,v)  = (bool)I(u,v,3);
    # [-2**15/128, 2**15/128] = [-256, 256]
    #flow_map_16bit = np.rint(flow_map*128 + 2**15).astype(np.uint16)
    _, h,w = flow.shape
    flow_map = np.rint(flow*128 + 2**15)
    flow_map = flow_map.astype(np.uint16).transpose(1,2,0)
    flow_map = np.concatenate((flow_map, np.zeros((h,w,1), dtype=np.uint16)), axis=-1)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = '{:06d}.png'.format(file_index)

    imageio.imwrite(save_dir / file_name, flow_map, format='PNG-FI')


def visualize_flow_image(save_dir:Path, flow: np.ndarray, file_index: int, method = "new"):
    method = method.lower()
    assert method in ["old", "new"]

    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = '{:06d}.png'.format(file_index)

    if method == "new":
        flow= flow.transpose(1, 2, 0)
        flow_img = flow_vis.flow_to_color(flow, convert_to_bgr = False)
        imageio.imwrite(save_dir / file_name, flow_img, format='PNG')
    else:
        visualize_optical_flow(flow, savepath = str(save_dir / file_name))


@torch.no_grad()
def generate_submission(model, save_path:str, visualize_flow = False, visualization_method = "new"):
    model.eval()
    test_dataset = DSECfull('test')

    bar = tqdm(test_dataset,total=len(test_dataset), ncols=60)
    bar.set_description('Test')

    save_path = Path(save_path)
    if visualize_flow:
        vis_path = save_path / "visualization"
        save_path = save_path / "submission"

    for voxel1, voxel2, seg, submission_coords in bar:
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda() 
        # seg = segmentation2rgb_19(seg[None])
        flow_pred, _ = model(voxel1, voxel2, seg)
        flow_pred = flow_pred[0].cpu()#[1,2,H,W]

        sequence, file_index = submission_coords
        save_dir = save_path / sequence

        save_flow_submission(save_dir, flow_pred.numpy(), file_index)

        if visualize_flow:
            visualize_flow_image(vis_path / sequence, flow_pred.numpy(), file_index, visualization_method)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser =ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to a saved checkpoint file (.pth)")
    parser.add_argument("-b", "--input_bins", type=int, default=15, help="Number of input bins")
    parser.add_argument("-s", "--save_path", type=str, default="./sbumission", help="Submission save path")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize optical flow")
    parser.add_argument("--old_vis_method", action="store_true", help="Use the old method of optical flow visualization")

    args = parser.parse_args()

    # Load model
    model = TMA(input_bins=args.input_bins)
    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.cuda()
    vis_method = "old" if args.old_vis_method else "new"

    generate_submission(model, args.save_path, args.visualize, vis_method)