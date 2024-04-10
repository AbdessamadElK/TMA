import numpy as np
import torch
import torch.utils.data as data

import random
import os
from pathlib import Path
import glob

from .augment import Augmentor
from .representation import VoxelGrid

class DSECfull(data.Dataset):
    def __init__(self, phase):
        assert phase in ["train", "trainval", "test"]

        self.init_seed = False
        self.phase = phase
        self.representation = VoxelGrid((15, 480, 640), normalize=True)
        self.files = []
        self.flows = []

        ### Please change the root to satisfy your data saving setting.
        root = 'datasets/dsec_full'
        if phase == 'train' or phase == 'trainval':
            self.root = os.path.join(root, 'trainval')
            self.augmentor = Augmentor(crop_size=[288, 384])
        else:
            self.root = os.path.join(root, 'test')


        self.files = glob.glob(os.path.join(self.root, '*', '*.npz'))
        self.files.sort()

        self.flows = glob.glob(os.path.join(self.root, '*', 'flow_*.npy'))
        self.flows.sort()

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
          

        #flow
        if self.phase == "train" or self.phase == "trainval":
            flow_16bit = np.load(self.flows[index])
            flow_map, valid2D = flow_16bit_to_float(flow_16bit)
        
       

        if self.phase == "test":
            # Include submission coordinates (seuence name, file index)
            file_path = Path(self.files[index])
            sequence_name = file_path.parent.name
            file_index = int(file_path.stem)
            submission_coords = (sequence_name, file_index)
            return events1, events2, submission_coords
        
        return events1, events2, flow_map, valid2D

    def __len__(self):
        return len(self.files)
    


class DataPrefetcher():
    def __init__(self, dataloader, phase):
        """
        This DataPrefetcher class as its name indicates, prefetches the raw data on the GPU. It performs data preprocessing on
        the GPU in order to speed up the training process.
        """
        assert phase in ["train", "trainval", "test"]
        self.dataloader = dataloader
        self.representation = VoxelGrid((15, 480, 640), normalize=True)
        self._len = len(dataloader)



    def prefetch(self):
        try:
            self.events1, self.events2, self.next_flow, self.next_valid = next(self.dl_iter)
        except StopIteration:
            self.events1, self.events2, self.next_flow, self.next_valid = [None, None, None, None]
            return
        
        # To cuda
        self.events1 = self.events1.cuda()
        self.events2 = self.events2.cuda()
        self.next_flow = self.next_flow.cuda()
        self.next_valid = self.next_valid.cuda()
        
        # Convert events to voxel grids
        x = self.events1[:, 0]
        y = self.events1[:, 1]
        t = self.events1[:, 2]
        p = self.events1[:, 3]
        self.next_voxel1 = self.events_to_voxel_grid(x, y, p, t).permute(1, 2, 0).numpy()

        x = self.events2[:, 0]
        y = self.events2[:, 1]
        t = self.events2[:, 2]
        p = self.events2[:, 3]
        self.next_voxel2 = self.events_to_voxel_grid(x, y, p, t).permute(1, 2, 0).numpy()

        # Apply data augmentation
        if self.phase == "train" or self.phase == "trainval":
           
            augmented = self.augmentor(self.next_voxel1, self.next_voxel2, self.next_flow, self.next_valid)

            self.next_voxel1, self.next_voxel2, self.next_flow, self.next_valid = augmented

            self.next_flow = torch.from_numpy(self.next_flow).permute(2, 0, 1).float()
            self.next_valid = torch.from_numpy(self.next_valid).float()

        self.next_voxel1 = torch.from_numpy(self.next_voxel1).permute(2, 0, 1).float()
        self.next_voxel2 = torch.from_numpy(self.next_voxel2).permute(2, 0, 1).float()
        
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
        return self._len

    def __iter__(self):
        self.dl_iter = iter(self.dataloader)
        self.prefetch()
        pass

    def __next__(self):
        voxel1 = self.next_voxel1
        voxel2 = self.next_voxel2
        flow_map = self.next_flow
        valid2D = self.next_valid

        if None in [voxel1, voxel2, flow_map, valid2D]:
            raise StopIteration
        
        self.prefetch()

        return voxel1, voxel2, flow_map, valid2D
    
    
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


def collate_fn(batch):
    #len(batch) = 6
    print(type(batch[0][0]))
    raise NotImplementedError


def make_data_loader(phase, batch_size, num_workers):
    dset = DSECfull(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn)
    prefetcher = DataPrefetcher(loader, phase = phase)
    return loader, prefetcher

if __name__ == '__main__':

    dset = DSECfull('test')
    print(len(dset))
    v1, v2, flow, valid = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape)