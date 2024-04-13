import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

from .augment import Augmentor
from .representation import VoxelGrid

class DSECsplit(data.Dataset):
    def __init__(self, phase):
        self.init_seed = False
        assert phase in ["train", "val"]
        self.phase = phase
        self.representation = VoxelGrid((15, 480, 640), normalize=True)
        self.files = []
        self.flows = []

        ### Please change the root to satisfy your data saving setting.
        root = 'datasets/dsec_split'
        if phase == 'train':
            self.root = os.path.join(root, 'train')
            self.augmentor = Augmentor(crop_size=[288, 384])
        else:
            self.root = os.path.join(root, 'val')


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
        flow_16bit = np.load(self.flows[index])
        flow_map, valid2D = flow_16bit_to_float(flow_16bit)

        return events1, events2, flow_map, valid2D
    
    def __len__(self):
        return len(self.files)
    

class DataPrefetcherSplit():
    def __init__(self, dataloader, phase, augment = False):
        """
        The DataPrefetcherSplit class takes a dataloader that provides raw data (raw events and optical flow),
        then It transforms the events into volumetric voxel grids on the GPU for faster performance, and applies
        data augmentation.
        """
        assert phase in ["train", "val"]
        self.dataloader = dataloader
        self.phase = phase
        self.representation = VoxelGrid((15, 480, 640), normalize=True)
        self._len = len(dataloader)

        self.augment = augment
        self.augmentor = Augmentor(crop_size=[288, 384])

    def prefetch(self):
        try:
            # Iterator returns a list of batch_size tuples, each contain (events1, events2, flow, valid)
            raw_batch = next(self.dl_iter)
    
        except StopIteration:
            self.next_voxel1 = None
            self.next_voxel2 = None
            self.next_flow = None
            self.next_valid = None
            return
        
        self.next_voxel1 = []
        self.next_voxel2 = []
        self.next_flow = []
        self.next_valid = []

        for elements in raw_batch:
            # Unpack data
            events1, events2, flow, valid = elements

            # To cuda
            events1 = torch.from_numpy(events1).cuda()
            events2 = torch.from_numpy(events2).cuda()
            
            # Convert events to voxel grids (on CUDA)
            x = events1[:, 0]
            y = events1[:, 1]
            t = events1[:, 2]
            p = events1[:, 3]
            voxel1 = self.events_to_voxel_grid(x, y, p, t).permute(1, 2, 0).cpu().numpy()

            x = events2[:, 0]
            y = events2[:, 1]
            t = events2[:, 2]
            p = events2[:, 3]
            voxel2 = self.events_to_voxel_grid(x, y, p, t).permute(1, 2, 0).cpu().numpy()

            # Apply data augmentation
            if self.phase == "train" and self.augment:
                voxel1, voxel2, flow, valid = self.augmentor(voxel1, voxel2, flow, valid)

            voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
            voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid).float()

            # Append to output
            self.next_voxel1.append(voxel1)
            self.next_voxel2.append(voxel2)
            self.next_flow.append(flow)
            self.next_valid.append(valid)

        # Convert outputs to torch tensor
        self.next_voxel1 = torch.stack(self.next_voxel1)
        self.next_voxel2 = torch.stack(self.next_voxel2)
        self.next_flow = torch.stack(self.next_flow)
        self.next_valid = torch.stack(self.next_valid)
        
    def events_to_voxel_grid(self, x, y, p, t):
        t = (t - t[0]).float()
        t = (t/t[-1])
        event_data_torch = {
            'p': p.float(),
            't': t,
            'x': x.float(),
            'y': y.float(),
            }
        return self.representation.convert(event_data_torch)
    
    def __len__(self):
        return self._len

    def __iter__(self):
        self.dl_iter = iter(self.dataloader)
        self.prefetch()
        return self

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


def make_data_loader(phase, batch_size, num_workers, data_augmentation = False):
    dset = DSECsplit(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch:batch)
    # The collate_fn returns the batch as is, and it will be dealt with later by the prefetcher  
    prefetcher = DataPrefetcherSplit(loader, phase, augment=data_augmentation)
    return loader, prefetcher

if __name__ == '__main__':

    dset = DSECsplit('test')
    print(len(dset))
    v1, v2, flow, valid = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape)