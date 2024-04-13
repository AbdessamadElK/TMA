import numpy as np
import torch
import torch.utils.data as data

import random
import os
from pathlib import Path
import glob

from .augment import Augmentor
from .representation import VoxelGrid

class DSECtest(data.Dataset):
    def __init__(self):
        self.init_seed = False
        self.files = []
        self.flows = []
        self.representation = VoxelGrid((15, 480, 640), normalize=True)

        ### Please change the root to satisfy your data saving setting.
        root = 'datasets/dsec_full'
        self.root = os.path.join(root, 'test')        

        self.files = glob.glob(os.path.join(self.root, '*', '*.npz'))
        self.files.sort()

        self.flows = glob.glob(os.path.join(self.root, '*', 'flow_*.npy'))
        self.flows.sort()

    
    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        #events
        events_file = np.load(self.files[index])
        events1 = events_file['events_prev']
        events2 = events_file['events_curr']

        
        # Convert events to voxel grids
        x = events1[:, 0]
        y = events1[:, 1]
        t = events1[:, 2]
        p = events1[:, 3]
        voxel1 = self.events_to_voxel_grid(x, y, p, t)

        x = events2[:, 0]
        y = events2[:, 1]
        t = events2[:, 2]
        p = events2[:, 3]
        voxel2 = self.events_to_voxel_grid(x, y, p, t)
        
        # Include submission coordinates (seuence name, file index)
        file_path = Path(self.files[index])
        sequence_name = file_path.parent.name
        file_index = int(file_path.stem)
        submission_coords = (sequence_name, file_index)

        return voxel1, voxel2, submission_coords

    def events_to_voxel_grid(self, x, y, p, t):
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
        return self.representation.convert(event_data_torch)

    def __len__(self):
        return len(self.files)
    
    
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


def make_data_loader(phase, batch_size, num_workers):
    dset = DSECtest(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,)
    return loader

if __name__ == '__main__':

    dset = DSECtest('test')
    print(len(dset))
    v1, v2, flow, valid = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape)