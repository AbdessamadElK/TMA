import numpy as np
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import shutil

import imageio.v2 as imageio
import cv2



def gen_dsec(split:str, resize_format:'tuple|None' = (480, 640)):
    assert split.lower() in ['train', 'trainval', 'test']
    if split == 'trainval' : split  = 'train'

    images_path = Path(f'E:/DSEC/{split}_images')

    if split == 'train':
        output_path = Path(f'C:/users/abdessamad/TMA/datasets/dsec_full/trainval')
        flow_path = Path('E:/DSEC/train_optical_flow')
        sequences = [seq.name for seq in flow_path.iterdir()]
    else:
        output_path = Path(f'C:/users/abdessamad/TMA/datasets/dsec_full/test')
        flow_path= Path('E:/DSEC/test_forward_optical_flow_timestamps')
        sequences = [path.stem for path in flow_path.iterdir()]


    for seq in sequences:

        # Get image list and timestamps
        # images_dir = images_path / seq / 'images' / 'left' / 'rectified'
        # images_dir = Path(f"D:/DSEC/{split}_left_images_distorted/{split}") / seq / 'images' / 'left' / 'distorted'
        images_dir = Path(f"E:/DSEC/{split}_images_warped") / seq / "warped"
        ts_file_img = images_path / seq / 'images' / 'timestamps.txt'
        timestamps_img = np.genfromtxt(str(ts_file_img), delimiter=',')
        img_list = sorted(glob(os.path.join(str(images_dir), '*.png')))
        assert timestamps_img.shape[0] == len(img_list)

        # Optical flow timestamps
        if split == 'train':
            ts_file_flow = flow_path / seq / 'flow' / 'forward_timestamps.txt'
            timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
            timestamps_flow  = np.append(timestamps_flow[:,0], timestamps_flow[-1,1])
        else:
            ts_file_flow = flow_path / f'{seq}.csv'
            timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
            timestamps_flow = timestamps_flow[:,:-1]
            timestamps_flow  = np.unique(timestamps_flow.flatten())


        output_dir = output_path / seq / 'images'
        if not os.path.exists(str(output_dir)):
            os.makedirs(output_dir)#

        for i in tqdm(range(len(timestamps_flow)), ncols=60):
            # Get image index
            t_curr = timestamps_flow[i]
            image_index = np.where(timestamps_img == t_curr)[0].item()
            save_path = output_dir / '{:06d}.png'.format(i)
          

            # Copy image to the corresponding output directory
            # img = imageio.imread(img_list[image_index])
            # if resize_format is not None:
            #     h, w = resize_format
            #     img = cv2.resize(img, (w, h))

            # imageio.imwrite(save_path, img)
            shutil.copy(img_list[image_index], save_path)

def verify_ts():
    split =  'train'

    images_path = Path(f'E:/DSEC/{split}_images')

    output_path = Path(f'C:/users/abdessamad/TMA/datasets/dsec_full/trainval')
    flow_path = Path('E:/DSEC/train_optical_flow')
    sequences = [seq.name for seq in flow_path.iterdir()]

    for seq in sequences:
        # if not seq == "zurich_city_02_d": continue
        # Get image list and timestamps
        images_dir = images_path / seq / 'images' / 'left' / 'rectified'
        # images_dir = Path(f"D:/DSEC/{split}_left_images_distorted/{split}") / seq / 'images' / 'left' / 'distorted'
        # images_dir = Path(f"E:/DSEC_distorted/{split}_images" / seq / "distorted" )
        ts_file_img = images_path / seq / 'images' / 'timestamps.txt'
        timestamps_img = np.genfromtxt(str(ts_file_img), delimiter=',')
        img_list = sorted(glob(os.path.join(str(images_dir), '*.png')))
        assert timestamps_img.shape[0] == len(img_list)

        # Optical flow timestamps
        ts_file_flow = flow_path / seq / 'flow' / 'forward_timestamps.txt'
        timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
        # timestamps_flow  = np.append(timestamps_flow[:,0], timestamps_flow[-1,1])
        timestamps_flow = timestamps_flow[:,1]

        count = 0

        for i in range(len(timestamps_img)):
            if timestamps_img[i] in timestamps_flow:
                count += 1

        print(seq, " : ", count, "/", len(timestamps_flow))


            
if __name__ == '__main__':
    gen_dsec('train')
    # verify_ts()