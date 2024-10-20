import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

from .augment import Augmentor

import imageio.v2 as imageio

class DSECsplit(data.Dataset):
    def __init__(self, phase):
        self.init_seed = False
        self.phase = phase
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

        self.images = glob.glob(os.path.join(self.root, '*', 'images', '*.png'))
        self.images.sort()

        self.segmentations = glob.glob(os.path.join(self.root, '*', 'segmentation', '*.png'))
        self.segmentations.sort()

    
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
        voxel1 = events_file['events_prev'].transpose(1, 2, 0)
        voxel2 = events_file['events_curr'].transpose(1, 2, 0)       

        #flow
        flow_16bit = np.load(self.flows[index])
        flow_map, valid2D = flow_16bit_to_float(flow_16bit)
        #image
        img = imageio.imread(self.images[index])

        #segmentation
        seg = imageio.imread(self.segmentations[index])

        # Ignore the bottom part of the segmentation
        seg[-20:,:] = 255
        
        if self.phase == 'train':
            voxel1, voxel2, flow_map, valid2D, img, seg = self.augmentor(voxel1, voxel2, flow_map, valid2D, img, seg)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        seg = torch.from_numpy(seg).float()
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
        flow_map = torch.from_numpy(flow_map).permute(2, 0, 1).float()
        valid2D = torch.from_numpy(valid2D).float()
        
        return voxel1, voxel2, flow_map, valid2D, img, seg
    
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
    dset = DSECsplit(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader

if __name__ == '__main__':

    dset = DSECsplit('test')
    print(len(dset))
    v1, v2, flow, valid = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape)