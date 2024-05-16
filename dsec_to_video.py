import skvideo.io as io

import numpy as np
import imageio.v2 as imageio

from pathlib import Path
from glob import glob

from tqdm import tqdm

from dataloader.dsec_full import DSECfull

from flow_vis import flow_to_color
from utils.visualization import visualize_optical_flow
from utils.visualization import segmentation2rgb_19

INPUT_PATH = "./datasets/dsec_full/trainval"
OUTPUT_PATH = "C:/users/abdessamad/TMA_DSEC_VIDEO"

OUTPUT_SINGLE = "C:/users/abdessamad/TMA_DSEC_VIDEO/no_aug.mp4"

INCLUDE_SEGMENTATION = False

def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D

def dsec_to_vid_separate():
    seq_dirs = Path(INPUT_PATH).iterdir()
    Path(OUTPUT_PATH).mkdir(parents = True, exist_ok = True)

    for seq_path in seq_dirs:
        # Locate all types of data
        flow_files = glob(str(seq_path / '*.npy'))
        image_files = glob(str(seq_path/'images'/'*.png'))
        seg_files = glob(str(seq_path/'segmentation'/'*.png'))

        flow_files.sort()
        image_files.sort()
        seg_files.sort()

        # Save path
        save_path = Path(OUTPUT_PATH) / f"{seq_path.name}.mp4"

        writer = io.FFmpegWriter(str(save_path), outputdict={"-pix_fmt": "yuv420p"})

        for idx in tqdm(range(len(flow_files)), desc=seq_path.name):
            rows = []

            # Optical flow visualization
            flow_16bit = np.load(flow_files[idx])
            flow_map, valid2D = flow_16bit_to_float(flow_16bit)

            mag = np.sum(flow_map**2, axis=2)
            mag = np.sqrt(mag)
            valid2D = (valid2D >= 0.5) & (mag <= 400)

            # flow ground truth
            flow_mask1 = flow_map.copy()
            flow_mask1[valid2D == 0] = np.array([0, 0])
            flow_vis, _ = visualize_optical_flow(flow_mask1.transpose(2, 0, 1))

            # image
            img = imageio.imread(image_files[idx])
        
            superpose = img.copy()
            superpose[valid2D] = 0.3 * superpose[valid2D] + 0.7 * flow_vis[valid2D] * 255 

            rows.append(np.hstack([flow_vis * 255, superpose]))

            # segmentation ground truth
            if INCLUDE_SEGMENTATION:
                seg = imageio.imread(seg_files[idx])
                seg_vis = segmentation2rgb_19(seg)

                seg_mask1 = seg_vis.copy()
                seg_mask1[valid2D == 0] = np.array([0, 0, 0])

                rows.append(np.hstack([seg_mask1, 0.6 * seg_vis + 0.4 * img]))

            frame = np.vstack(rows)

            writer.writeFrame(frame.astype('uint8'))

        writer.close()


def dsec_to_vid_single():
    "Outputs the DSEC dataset to a single video using the dataloader"
    dataset = DSECfull('trainval', crop=False, flip=False, spatial_aug=False)
    writer = io.FFmpegWriter(OUTPUT_SINGLE, outputdict={"-pix_fmt": "yuv420p"})

    for (_, _, flow_gt, valid2D, img, seg_gt) in tqdm(dataset, total = len(dataset)):
        
        row1 = []
        # flow_gt = flow_gt.numpy().transpose(1, 2, 0)
        # flow_gt[valid2D == 0] = 0
        # flow_gt_vis, _ = visualize_optical_flow(flow_gt.transpose(2, 0, 1))
        flow_gt_vis, _ = visualize_optical_flow(flow_gt.numpy())
        row1.append(flow_gt_vis * 255)

        flowx = flow_gt[0,:,:] * flow_gt[1,:,:] 
        flow_valid = flowx != 0

        

        img = img.numpy().transpose(1, 2, 0)
        superposed = img.copy()
        superposed[flow_valid] = 0.3 * superposed[flow_valid] + 0.7 * flow_gt_vis[flow_valid] * 255
        row1.append(superposed)

        if not INCLUDE_SEGMENTATION:
            writer.writeFrame(np.hstack(row1).astype('uint8'))
            continue
        
        row2 = []
        seg_gt = seg_gt.numpy()       
        seg_gt = segmentation2rgb_19(seg_gt)
        row2.append(seg_gt)

        row2.append(0.4 * img + 0.6 * seg_gt)

        # Write a frame
        frame = np.vstack([np.hstack(row1), np.hstack(row2)])
        writer.writeFrame(frame.astype('uint8'))

    writer.close()


if __name__ == "__main__":
    dsec_to_vid_separate()
