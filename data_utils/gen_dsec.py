from event_slicer import EventSlicer
import h5py
import hdf5plugin
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio

import torch
import cv2

from pathlib import Path

from representation import VoxelGrid

RESOLUTION = (480, 640)
TEMPORAL_BINS = 15

representation = VoxelGrid((TEMPORAL_BINS, *RESOLUTION), normalize=True)

def rectify_and_write(rect_map, events_curr, events_prev, output_dir, idx):
    #rectify events and convert to voxel grids
    p = events_curr['p']
    t = events_curr['t']
    x = events_curr['x']
    y = events_curr['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    events1 = events_to_voxel_grid(x_rect, y_rect, p, t)

    p = events_prev['p']
    t = events_prev['t']
    x = events_prev['x']
    y = events_prev['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    events0 = events_to_voxel_grid(x_rect, y_rect, p, t)

    #write events
    output_name = os.path.join(output_dir, '{:06d}'.format(idx))
    if os.path.exists(output_name):
        return
    np.savez(output_name, events_prev=events0, events_curr=events1)

def events_to_voxel_grid(x, y, p, t):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
        }
    return representation.convert(event_data_torch)

def gen_dsec(dsec_path:Path, split = 'train', images = True, distorted = False, seg_path:'Path|None' = None):
    assert dsec_path.is_dir()
    assert split.lower() in ['train', 'trainval', 'test']
    if split == 'trainval' : split  = 'train'

    event_path = dsec_path / f'{split}_events'

    if images:
        images_path = dsec_path / f'{split}_images'
        images_distorted_path = dsec_path / f'{split}_images_distorted'
    
    if split == 'train':
        output_root = Path(f'datasets/dsec_full/trainval')
        flow_path = dsec_path / f'{split}_optical_flow'
        sequences = [seq.name for seq in flow_path.iterdir()]
    else:
        output_root = Path(f'datasets/dsec_full/test')
        flow_path= Path('E:/DSEC/test_forward_optical_flow_timestamps')
        sequences = [path.stem for path in flow_path.iterdir()]

    for seq in sequences:
        # Get event files
        event_h5 = event_path / seq / 'events/left/events.h5'
        rectify_h5 = event_path / seq / 'events/left/rectify_map.h5'

        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]

        # Optical flow and flow timestamps
        if split == 'train':
            ts_file_flow = flow_path / seq / 'flow/forward_timestamps.txt'
            flow_dir = flow_path / seq / 'flow/forward'
            flow_list = sorted(flow_dir.glob("*.png")) 
            timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
            assert timestamps_flow.shape[0]== len(flow_list)
            # timestamps_flow  = np.append(timestamps_flow[:,0], timestamps_flow[-1,1])
        else:
            ts_file_flow = flow_path / f'{seq}.csv'
            timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
            timestamps_flow, indexs = timestamps_flow[:,:2], timestamps_flow[:,2]
        
        # Images and image timestamps
        if images:
            ts_file_img = images_path / seq / 'images' / 'timestamps.txt'
            if not distorted:
                img_dir = images_path / seq / 'images/left/rectified'
            else:
                img_dir = images_distorted_path / seq / 'distorted'

            timestamps_img = np.genfromtxt(ts_file_img, delimiter=',')
            img_list = sorted(img_dir.glob("*.png"))
            assert timestamps_img.shape[0] == len(img_list)

        # Semantic segmentation files
        if seg_path is not None:
            seg_dir = seg_path / seq
            seg_list = sorted(seg_dir.glob("*.png"))

        output_dir = output_root / seq
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(len(timestamps_flow)), ncols=60, desc=seq):
            #current events
            t_curr = timestamps_flow[i,0]
            t_next = timestamps_flow[i,1]
            events_curr = slicer.get_events(t_curr, t_next)
            if events_curr == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for current condition!')
                continue
            

            # previous events
            dt = 100 * 1000#us
            t_prev = t_curr - dt
            events_prev = slicer.get_events(t_prev, t_curr)
            if events_prev == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for previous condition!')
                continue

            save_idx = i if split == "train" else int(indexs[i])
            rectify_and_write(rectify_map, events_curr, events_prev, output_dir, save_idx)

            # optical flow
            if split == "train":
                flow_16bit = imageio.imread(flow_list[i], format='PNG-FI')
                np.save(os.path.join(output_dir, 'flow_{:06d}.npy'.format(save_idx)), flow_16bit)

            # image
            if images:
                image_index = np.where(timestamps_img == t_curr)[0].item()
                output_img = output_root / seq / 'images'
                output_img.mkdir(parents=True, exist_ok=True)

                img = imageio.imread(img_list[image_index])
                h, w = RESOLUTION
                if not (img.shape[0] == h and img.shape[1] == w):
                    img = cv2.resize(img, (w, h))

                save_path = output_img / '{:06d}.png'.format(save_idx)
                imageio.imwrite(save_path, img)

            # segmentation
            if seg_path is not None:
                output_seg = output_root / seq / 'segmentation'
                output_seg.mkdir(parents=True, exist_ok=True)

                seg = imageio.imread(seg_list[i])
                if not (seg.shape[0] == h and seg.shape[1] == w):
                    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

                save_path = output_seg / '{:06d}.png'.format(save_idx)
                imageio.imwrite(save_path, seg)


        h5f.close()



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dsec", type=Path, required=True, help="Path to the original DSEC dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to generate [[train]/test/all]")
    parser.add_argument("-i", "--images", default=False, action="store_true", help="Include images")
    parser.add_argument("-d", "--distorted", default=False, action="store_true", help="Use distorted images (towards events) instead of rectified ones")
    parser.add_argument("--segmentation", type=str, default="", help="Path to semantic segmentation")

    args = parser.parse_args()

    seg_path = None if args.segmentation == "" else Path(args.segmentation)
    splits = ['train' / 'test'] if args.split == "all" else [args.split]

    for split in splits:
        print(f"Generating TMA version of DSEC {split} split :")
        gen_dsec(args.dsec, split, args.images, args.distorted, seg_path)