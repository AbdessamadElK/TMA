import skvideo.io as io

import numpy as np
import imageio.v2 as imageio

from pathlib import Path
from glob import glob

from tqdm import tqdm

from dataloader.dsec_full import DSECfull

from data_utils.event_slicer import EventSlicer
import h5py
import hdf5plugin

import torch

from flow_vis import flow_to_color
from utils.visualization import visualize_optical_flow, segmentation2rgb_19, events_to_event_image

import cv2

# Change according to your setting
INPUT_PATH = "./datasets/dsec_full/trainval"
OUTPUT_PATH = "C:/users/abdessamad/TMA_DSEC_VIDEO/warped_images"

OUTPUT_SINGLE = "C:/users/abdessamad/TMA_DSEC_VIDEO/no_aug.mp4"

INCLUDE_SEGMENTATION = True

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
        if not seq_path.is_dir():
            continue
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
                seg_vis = segmentation2rgb_19(seg[None])[0]

                # seg_mask1 = seg_vis.copy()
                # seg_mask1[valid2D == 0] = np.array([0, 0, 0])

                rows.append(np.hstack([seg_vis, 0.6 * seg_vis + 0.4 * img]))

            frame = np.vstack(rows)

            writer.writeFrame(frame.astype('uint8'))

        writer.close()

def dsec_to_vid_test(input_path:Path, output_path:Path, segmentation = False):
    assert input_path.is_dir()
    seq_dirs = input_path.iterdir()
    Path(output_path).mkdir(parents = True, exist_ok = True)

    for seq_path in seq_dirs:
        if not seq_path.is_dir():
            continue
        # Locate all types of data
        image_files = glob(str(seq_path/'images'/'*.png'))
        image_files.sort()

        if segmentation:
            seg_files = glob(str(seq_path/'segmentation'/'*.png'))
            seg_files.sort()

        # Save path
        save_path = Path(output_path) / f"{seq_path.name}.mp4"

        writer = io.FFmpegWriter(str(save_path), outputdict={"-pix_fmt": "yuv420p"})

        for idx in tqdm(range(len(image_files)), desc=seq_path.name):
            # image
            img = imageio.imread(image_files[idx])
            row = [img]
        
            # segmentation ground truth
            if segmentation:
                seg = imageio.imread(seg_files[idx])
                seg_vis = segmentation2rgb_19(seg)

                # seg_mask1 = seg_vis.copy()
                # seg_mask1[valid2D == 0] = np.array([0, 0, 0])

                row = row + [0.6 * seg_vis + 0.4 * img, seg_vis]

                frame = np.hstack(row)
                writer.writeFrame(frame.astype('uint8'))

                continue
            
            writer.writeFrame(img.astype('uint8'))

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

import os
def verify_alignement(sequence = 'all'):
    OUTPUT = Path("C:/users/abdessamad/DSEC_ALIGNMENT_ZURICH01A/DISTORT_IMAGES_ALL")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    WARPED_SAVE_DIR = Path("E:/DSEC/warped_images")

    DELTA_TIME_MS = 100
    DELTA_TIME_US = DELTA_TIME_MS * 1000

    root = Path("E:/DSEC/")
    # root = Path("D:/DSEC_zurich01a")

    if sequence.lower() == 'all':
        flow_sequences = [folder.name for folder in (root / "train_optical_flow").iterdir()]
    else:
        assert (root / "train_optical_flow" / sequence).is_dir()
        flow_sequences = [sequence]

    for seq in flow_sequences:
        # Get image list and timestamps
        images_path = root / "train_images"
        images_dir = images_path / seq / 'images' / 'left' / 'rectified'
        images_dir_dist = root /"train_images_distorted" / seq / "distorted"
        # images_dir = Path("D:/DSEC/train_left_images_distorted/train") / seq / 'images' / 'left' / 'distorted'
        # images_dir = Path("E://DSEC-Lidar/train_images") / seq / "distorted"
        ts_file_img = images_path / seq / 'images' / 'timestamps.txt'
        timestamps_img = np.genfromtxt(str(ts_file_img), delimiter=',')
        img_list = sorted(glob(os.path.join(str(images_dir), '*.png')))
        assert timestamps_img.shape[0] == len(img_list)

        # Optical flow list and timestamps
        flow_path = root / 'train_optical_flow'
        ts_file_flow = flow_path / seq / 'flow' / 'forward_timestamps.txt'
        timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')

        flow_dir = flow_path / seq / "flow/forward"
        flow_list = sorted(glob(os.path.join(flow_dir, '*.png')))

        # Events files
        events_path = root / 'train_events'
        event_h5 = events_path / seq / 'events/left' / 'events.h5'
        rectify_h5 = events_path/  seq / 'events/left' / 'rectify_map.h5'

        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]

        writer = io.FFmpegWriter(OUTPUT / f"{seq}.mp4", outputdict={"-pix_fmt": "yuv420p"})
        save_dir = WARPED_SAVE_DIR / seq / "warped"
        save_dir.mkdir(parents = True, exist_ok = True)

        try:
            for i in tqdm(range(len(flow_list)), ncols=60, desc=seq):
                    # Get image index
                    t_start, t_end = timestamps_flow[i]
                    # image_index = np.where(timestamps_img == t_start)[0].item()          
                    
                    # img = img_list[image_index]
                    # img = imageio.imread(img)

                    # Instead, get image by file name
                    flow_file = flow_list[i]
                    image_file_name = images_dir / Path(flow_file).name
                    img = imageio.imread(image_file_name)

                    # Get optical flow visualization
                    flow_16bit = imageio.imread(flow_list[i], format='PNG-FI')
                    flow_map, valid2D = flow_16bit_to_float(flow_16bit)


                    h, w, _ = flow_map.shape
                    if not (img.shape[0] == h and img.shape[1] == w):
                        img = cv2.resize(img, (w, h))

                    mag = np.sum(flow_map**2, axis=2)
                    mag = np.sqrt(mag)
                    valid2D = (valid2D >= 0.5) & (mag <= 400)

                    # flow ground truth
                    flow_mask1 = flow_map.copy()
                    flow_mask1[valid2D == 0] = np.array([0, 0])
                    flow_vis, _ = visualize_optical_flow(flow_mask1.transpose(2, 0, 1))

                    # get raw events (only half of the volume for better visualization)
                    events = slicer.get_events(t_start, t_start + DELTA_TIME_US)
                    t = events["t"]
                    x = events["x"]
                    y = events["y"]
                    p = events["p"]
                    p = p * 2.0 - 1.0

                    events = np.vstack([t, x, y, p]).transpose()
                    
                    # get rectified events
                    xy_rect = rectify_map[y, x]
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]

                    events_rect = np.vstack([t, x_rect, y_rect, p]).transpose()

                    # Also get distored image
                    flow_file = flow_list[i]
                    image_file_name = images_dir_dist / Path(flow_file).name
                    img_dist = imageio.imread(image_file_name)


                    frame_rows = []
                    for image in [img_dist]:
                                
                        row = []

                        # Warp the images to the rectified events domain
                        warped_img = np.zeros(image.shape)
                        h, w, _ = image.shape
                        for y in range(h):
                            for x in range(w):
                                new_x, new_y = rectify_map[y, x]
                                if new_x <= 0 or new_y <= 0:
                                    continue
                                
                                if new_x >= w or new_y >= h:
                                    continue
                                
                                new_x = int(new_x)
                                new_y = int(new_y)

                                warped_img[new_y, new_x, :] = image[y, x, :]

                        # Save the warped image
                        imageio.imwrite(save_dir / Path(flow_file).name, warped_img.astype('uint8'))
                        

                        row.append(warped_img.copy())
                        image = warped_img

                        # Add events on image
                        
                        # events_vis = events_to_event_image(events, h, w, background = torch.full((3, h, w), 0).byte()).numpy().transpose(1, 2, 0)
                        # events_mask = events_vis != [0,0,0]
                        # events_img = image.copy()
                        # events_img[events_mask] = 0.3 * events_img[events_mask] + 0.7 * events_vis[events_mask]
                        # row.append(events_img)
                        
                        # Rectified events
                        events_vis_rect = events_to_event_image(events_rect, h, w, background = torch.full((3, h, w), 0).byte()).numpy().transpose(1, 2, 0)
                        events_mask_rect = events_vis_rect != [0,0,0]
                        events_img_rect = image.copy()
                        events_img_rect[events_mask_rect] = 0.3 * events_img_rect[events_mask_rect] + 0.7 * events_vis_rect[events_mask_rect]

                        row.append(events_img_rect)

                        # Flow on image
                        flow_img = image.copy()
                        flow_img[valid2D] = 0.3 * flow_img[valid2D] + 0.7 * flow_vis[valid2D] * 255
                        row.append(flow_img)


                        frame_rows.append(np.hstack(row))


                    frame = np.vstack(frame_rows)
                    writer.writeFrame(frame.astype('uint8'))

        except KeyboardInterrupt:
            print("Quitting...")
            writer.close()

        writer.close()


def interpolate_warped_images():
    OUTPUT = Path("C:/users/abdessamad/DSEC_WARP_INTER_IMAGES")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    WARPED_SAVE_DIR = Path("E:/DSEC/warped_images")

    DELTA_TIME_MS = 100
    DELTA_TIME_US = DELTA_TIME_MS * 1000

    root = Path("E:/DSEC/")
    # root = Path("D:/DSEC_zurich01a")

    flow_sequences = [folder.name for folder in (root / "train_optical_flow").iterdir()]
    
   
    for seq in flow_sequences:
        # Get image list and timestamps
        images_path = root / "train_images"
        images_dir = images_path / seq / 'images' / 'left' / 'rectified'
        images_dir_dist = root /"train_images_warped" / seq / "warped"
        # images_dir = Path("D:/DSEC/train_left_images_distorted/train") / seq / 'images' / 'left' / 'distorted'
        # images_dir = Path("E://DSEC-Lidar/train_images") / seq / "distorted"
        ts_file_img = images_path / seq / 'images' / 'timestamps.txt'
        timestamps_img = np.genfromtxt(str(ts_file_img), delimiter=',')
        img_list = sorted(glob(os.path.join(str(images_dir), '*.png')))
        assert timestamps_img.shape[0] == len(img_list)

        # Optical flow list and timestamps
        flow_path = root / 'train_optical_flow'
        ts_file_flow = flow_path / seq / 'flow' / 'forward_timestamps.txt'
        timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')

        flow_dir = flow_path / seq / "flow/forward"
        flow_list = sorted(glob(os.path.join(flow_dir, '*.png')))

        # Events files
        events_path = root / 'train_events'
        event_h5 = events_path / seq / 'events/left' / 'events.h5'
        rectify_h5 = events_path/  seq / 'events/left' / 'rectify_map.h5'

        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]

        writer = io.FFmpegWriter(OUTPUT / f"{seq}.mp4", outputdict={"-pix_fmt": "yuv420p"})
        save_dir = WARPED_SAVE_DIR / seq / "warped"
        save_dir.mkdir(parents = True, exist_ok = True)

        for i in tqdm(range(len(flow_list)), ncols=60, desc=seq):
                # Get image index
                t_start, t_end = timestamps_flow[i]
                # image_index = np.where(timestamps_img == t_start)[0].item()          
                
                # img = img_list[image_index]
                # img = imageio.imread(img)

                # Instead, get image by file name
                flow_file = flow_list[i]
                image_file_name = images_dir / Path(flow_file).name
                img = imageio.imread(image_file_name)

                # Get optical flow visualization
                flow_16bit = imageio.imread(flow_list[i], format='PNG-FI')
                flow_map, valid2D = flow_16bit_to_float(flow_16bit)


                h, w, _ = flow_map.shape
                if not (img.shape[0] == h and img.shape[1] == w):
                    img = cv2.resize(img, (w, h))

                mag = np.sum(flow_map**2, axis=2)
                mag = np.sqrt(mag)
                valid2D = (valid2D >= 0.5) & (mag <= 400)

                # flow ground truth
                flow_mask1 = flow_map.copy()
                flow_mask1[valid2D == 0] = np.array([0, 0])
                flow_vis, _ = visualize_optical_flow(flow_mask1.transpose(2, 0, 1))

                # get raw events (only half of the volume for better visualization)
                events = slicer.get_events(t_start, t_start + DELTA_TIME_US)
                t = events["t"]
                x = events["x"]
                y = events["y"]
                p = events["p"]
                p = p * 2.0 - 1.0

                events = np.vstack([t, x, y, p]).transpose()
                
                # get rectified events
                xy_rect = rectify_map[y, x]
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                events_rect = np.vstack([t, x_rect, y_rect, p]).transpose()

                # Also get a warped image
                flow_file = flow_list[i]
                image_file_name = images_dir_dist / Path(flow_file).name
                warped_img = imageio.imread(image_file_name)


                frame_rows = []                            
                row = []

                # Interpolate the warped image
                # Get empty pixels (pixels that will not be filled after rectification)
                gray_image = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)  # Create mask for inpainting

                # Perform inpainting using Telea's method
                interpolated_image = cv2.inpaint(warped_img, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)

                # Save the warped image
                imageio.imwrite(save_dir / Path(flow_file).name, interpolated_image.astype('uint8'))
                
                image = interpolated_image.copy()
                row.append(image)

                # Add events on image
                
                # events_vis = events_to_event_image(events, h, w, background = torch.full((3, h, w), 0).byte()).numpy().transpose(1, 2, 0)
                # events_mask = events_vis != [0,0,0]
                # events_img = image.copy()
                # events_img[events_mask] = 0.3 * events_img[events_mask] + 0.7 * events_vis[events_mask]
                # row.append(events_img)
                
                # Rectified events
                events_vis_rect = events_to_event_image(events_rect, h, w, background = torch.full((3, h, w), 0).byte()).numpy().transpose(1, 2, 0)
                events_mask_rect = events_vis_rect != [0,0,0]
                events_img_rect = image.copy()
                events_img_rect[events_mask_rect] = 0.3 * events_img_rect[events_mask_rect] + 0.7 * events_vis_rect[events_mask_rect]

                row.append(events_img_rect)

                # Flow on image
                flow_img = image.copy()
                flow_img[valid2D] = 0.3 * flow_img[valid2D] + 0.7 * flow_vis[valid2D] * 255
                row.append(flow_img)


                frame_rows.append(np.hstack(row))


                frame = np.vstack(frame_rows)
                writer.writeFrame(frame.astype('uint8'))

import random
from matplotlib import pyplot as plt
def interpolate_images():
    seq = 'zurich_city_01_a'
    root = Path("E:/DSEC")
    images_dir = root / "train_images_warped" / seq / "warped"

    # Events files
    events_path = root / 'train_events'
    rectify_h5 = events_path/  seq / 'events/left' / 'rectify_map.h5'


    with h5py.File(rectify_h5, 'r') as h5_rect:
        rectify_map = h5_rect['rectify_map'][()]

    images = list(images_dir.iterdir())

    img = imageio.imread(random.choice(images))

    # Get empty pixels (pixels that will not be filled after rectification)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)  # Create mask for inpainting

    # Perform inpainting using Telea's method
    inpainted_image = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)

    # Display the result
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title('Inpainted Image')
    plt.imshow(inpainted_image)

    plt.show()

    # print(img.shape)
    
    # img[img==[0,0,0]]

    return

    # dummy = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # rect_map_y = [[1, 0, -1],
    #               [1, 1, 1],
    #               [2, 2, 2]]

    

    # rect_map_x = [[1, 2, 3],
    #               [-1, 0, 3],
    #               [-1, 2, 3]]

    

    # rect_map = np.dstack([rect_map_x, rect_map_y])

    # print(rect_map, end="\n\n")

    # w, h, _ = rect_map.shape
    # m = np.meshgrid(np.arange(w), np.arange(h))
    # m = np.reshape(np.dstack(m), (3, 3, 2))
    # empty = np.setdiff1d(m, rect_map)

    # print(empty)

    # m = np.reshape(np.array(m), (3, 3, 2))
    # print(m)



if __name__ == "__main__":
    TRAIN_INPUT = Path("C:/users/abdessamad/TMA/datasets/dsec_full/trainval")
    TEST_INPUT = Path("C:/users/abdessamad/TMA/datasets/dsec_full/test")

    TRAIN_OUTPUT = Path("C:/users/abdessamad/TMA_DSEC_VIDEO/train")
    TEST_OUTPUT = Path("C:/users/abdessamad/TMA_DSEC_VIDEO/test")

    dsec_to_vid_separate()
    # verify_alignement()
    # dsec_to_vid_test(TEST_INPUT, Path("C:/users/abdessamad/TMA_DSEC_VIDEO_TEST"), segmentation=True)

    # interpolate_warped_images()