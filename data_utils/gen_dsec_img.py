import numpy as np
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    images_path = Path('D:/DSEC_flow/train')
    flow_path = Path('D:/DSEC_flow/train')
    output_path = Path('C:/users/abdessamad/TMA/datasets/dsec_full/trainval')
    sequences = [seq.name for seq in images_path.iterdir()]

    for seq in sequences:

        # Get image list and timestamps
        images_dir = images_path / seq / 'images_left' / 'rectified'
        ts_file_img = images_path / seq / 'images_timestamps.txt'
        timestamps_img = np.genfromtxt(str(ts_file_img), delimiter=',')
        img_list = sorted(glob(os.path.join(str(images_dir), '*.png')))
        assert timestamps_img.shape[0] == len(img_list)

        # Optical flow timestamps
        ts_file_flow = flow_path / seq / 'flow_forward_timestamps.txt'
        timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
        timestamps_flow  = np.unique(timestamps_flow.flatten())

        output_dir = output_path / seq
        if not os.path.exists(str(output_dir)):
            os.makedirs(output_dir)

        for i in tqdm(range(len(timestamps_flow)), ncols=60):
            # Get image index
            t_curr = timestamps_flow[i]
            image_index = np.where(timestamps_img == t_curr)

            # Copy image to the corresponding output directory
            save_path = output_dir / 'images' / '{:06d}.png'.format(i)
            shutil.copy(img_list[i], save_path)
