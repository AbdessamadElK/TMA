from event_slicer import EventSlicer
import h5py
import hdf5plugin
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import imageio

if __name__ == '__main__':
    images_path = 'C:/users/abdessamad/DSEC_TMA/images'
    event_path = 'C:/users/abdessamad/DSEC_TMA/train_events' 
    flow_path = 'C:/users/abdessamad/DSEC_TMA/train_optical_flow'

    sequences = os.listdir(flow_path)
    for seq in sequences:
        event_h5 = os.path.join(event_path, seq, 'events/left', 'events.h5')
        rectify_h5 = os.path.join(event_path, seq, 'events/left', 'rectify_map.h5')

        ts_file = os.path.join(flow_path, seq, 'flow', 'forward_timestamps.txt')
        flow_dir = os.path.join(flow_path, seq, 'flow', 'forward') 
        flow_list = sorted(glob(os.path.join(flow_dir, '*.png')))
        timestamps = np.genfromtxt(ts_file, delimiter=',')
        assert timestamps.shape[0]== len(flow_list)
        
        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]

        output_dir = os.path.join('datasets/dsec_full/trainval', seq)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(len(timestamps)), ncols=60):
            #current
            t_curr = timestamps[i,0]
            t_next = timestamps[i,1]
            events_curr = slicer.get_events(t_curr, t_next)
            if events_curr == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for current condition!')
                continue

            # previous 
            dt = 100 * 1000#us
            t_prev = t_curr - dt
            events_prev = slicer.get_events(t_prev, t_curr)
            if events_prev == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for previous condition!')
                continue


            flow_16bit = imageio.imread(flow_list[i], format='PNG-FI')
            np.save(os.path.join(output_dir, 'flow_{:06d}.npy'.format(i)), flow_16bit)
        h5f.close()