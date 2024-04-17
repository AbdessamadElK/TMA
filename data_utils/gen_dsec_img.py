import numpy as np
import os
from glob import glob
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    images_path = 'D:/DSEC_flow/train'
    flow_path = 'D:/DSEC_flow/train'
    output_path = 'C:/users/abdessamad/TMA/datasets/dsec_full/trainval'
    sequences = os.listdir(images_path)

    for seq in sequences:

        # Get image list and timestamps
        images_dir = os.path.join(images_path, seq, 'images_left/rectified')
        ts_file_img = os.path.join(images_path, seq, 'images_timestamps.txt')
        timestamps_img = np.genfromtxt(ts_file_img, delimiter=',')
        img_list = sorted(glob(os.path.join(images_dir, '*.png')))
        assert timestamps_img.shape[0] == len(img_list)

        # Optical flow timestamps
        ts_file_flow = os.path.join(flow_path, seq, 'flow_forward_timestamps.txt')
        timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
        timestamps_flow  = np.unique(timestamps_flow.flatten())

        output_dir = os.path.join(output_path, seq)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(len(timestamps_flow)), ncols=60):
            # Get image index
            t_curr = timestamps_flow[i]
            image_index = np.where(timestamps_img == t_curr)

            # Copy image to the corresponding output directory
            save_path = os.path.join(output_dir, 'images', '{:06d}.png'.format(i))
            shutil.copy(img_list[i], save_path)
