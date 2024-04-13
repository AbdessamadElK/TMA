import sys
sys.path.append('model')
import time
import os
import random
from tqdm import tqdm
import wandb
import torch
import numpy as np
from utils.file_utils import get_logger
from dataloader.dsec_full import make_data_loader

####Important####
from model.TMA import TMA
####Important####

from datetime import datetime
from torchvision.transforms import v2
from flow_vis import flow_to_color

MAX_FLOW = 400
SUM_FREQ = 100
VIS_FREQ = 5000
SAVE_FREQ = 10000

CROP_HEIGTH = 288
CROP_WIDTH = 384

# Half precision
#torch.set_default_dtype(torch.float16)

class Loss_Tracker:
    def __init__(self, wandb):
        self.running_loss = {}
        self.total_steps = 0
        self.wandb = wandb
    def push(self, metrics):
        self.total_steps += 1
        
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            
            self.running_loss[key] += metrics[key]
        
        if self.total_steps % SUM_FREQ == 0:
            if self.wandb:
                wandb.log({'EPE': self.running_loss['epe']/SUM_FREQ}, step=self.total_steps)
            self.running_loss = {}
        
    def state_dict(self):
        return {"running_loss":self.running_loss,
                "total_steps":self.total_steps,
                "wandb":self.wandb}
    
    def load_state_dict(self, state_dict:dict):
        keys = ["running_loss", "total_steps", "wandb"]
        for key in keys:
            self.__setattr__(key, state_dict[key])

            

class Trainer:
    def __init__(self, args):
        self.args = args

        self.model = TMA(input_bins=15)
        self.model = self.model.cuda()

        self.date_label = datetime.now().strftime("%Y-%m-%d")
        self.save_path = os.path.join(args.checkpoint_dir, self.date_label)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        #Data augmentation
        self.augment = args.augment

        #Loader
        self.train_loader, self.train_prefetcher = make_data_loader('trainval', args.batch_size, args.num_workers, data_augmentation=self.augment)
        print('train_loader done!')

        #Optimizer and scheduler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=args.lr,
            total_steps=args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='linear')
        #Logger
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.writer = get_logger(os.path.join(self.checkpoint_dir, self.date_label, 'train.log'))
        self.tracker = Loss_Tracker(args.wandb)

        
        #Loading checkpoint
        self.old_ckpt_path = args.restore_ckpt_path
        self.continue_training = args.continue_training
        self.previous_step = None

        if not self.old_ckpt_path == "":
            if os.path.isfile(self.old_ckpt_path):
                self.previous_step = self.load_ckpt(self.old_ckpt_path, self.continue_training)
                self.writer.info(f"Loaded the checkpoint at '{self.old_ckpt_path}'.")
                if self.continue_training:
                    self.writer.info("Also loaded parameters for continuous learning.")
            else:
                print("Couldn't find a checkpoint file at '{}'".format(self.old_ckpt_path))

        self.writer.info('====A NEW TRAINING PROCESS====')

    def train(self):
        # self.writer.info(self.model)
        self.writer.info(self.args)
        self.model.train()
        
        vis_steps = 0
        total_steps = 0 if not self.continue_training else self.previous_step

        keep_training = True
        while keep_training:

            bar = tqdm(enumerate(self.train_prefetcher),total=len(self.train_prefetcher), ncols=60)
            for index, (voxel1, voxel2, flow_map, valid2D) in bar:
                # voxel1, voxel2, flow_map, valid2D = self.apply_transforms(data_items)
                self.optimizer.zero_grad()
                flow_preds = self.model(voxel1.cuda(), voxel2.cuda())
                flow_loss, loss_metrics = sequence_loss(flow_preds, flow_map.cuda(), valid2D.cuda(), self.args.weight, MAX_FLOW)
                
                flow_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                bar.set_description(f'Step: {total_steps}/{self.args.num_steps}')
                self.tracker.push(loss_metrics)
                total_steps += 1
                if total_steps and total_steps % VIS_FREQ == 0 and self.args.wandb:
                    vis_steps += 1
                    with torch.no_grad():
                        #flow_preds: (12, 6, 2, h, w)
                        flow_sample = flow_preds[-1][0]
                        visualization = flow_to_color(flow_sample.cpu().numpy().transpose(1, 2, 0), convert_to_bgr = False)
                        wandb.log({'Optical Flow': wandb.Image(visualization, caption=f"Visualization {vis_steps}")})

                if total_steps and total_steps % SAVE_FREQ == 0:
                    # Save model checkpoint
                    ckpt = os.path.join(self.save_path, f'checkpoint_{total_steps}')

                    # Save checkpoint with parameters for continuous training
                    params_state = {"step":total_steps,
                            "model":self.model.state_dict(),
                            "optimizer":self.optimizer.state_dict(),
                            "scheduler":self.scheduler.state_dict(),
                            "loss_tracker":self.tracker.state_dict()}

                    torch.save(params_state, ckpt)

                if total_steps > self.args.num_steps:
                    keep_training = False
                    break
            
            time.sleep(0.03)
        
        # Save the final checkpoint
        model_ckpt_path = os.path.join(self.save_path, "checkpoint.pth")
        torch.save(self.model.state_dict(), model_ckpt_path)

        return model_ckpt_path
    
    def load_ckpt(self, ckpt_path:str, continuous = False):
        if os.path.isfile(ckpt_path):
            # Load the model
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint["model"], strict=False)

            # Load training params
            if continuous:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                self.tracker.load_state_dict(checkpoint["loss_tracker"])

                return checkpoint["step"]
        else:
            print("Warning : No checkpoint was found at '{}'".format(ckpt_path))

    def apply_transforms(self, data_items):
        transformed_items = data_items
        if self.crop:
            crop_size = (CROP_HEIGTH, CROP_WIDTH)
            rand_crop = v2.RandomCrop(crop_size)
            i, j, h, w = rand_crop.get_params(data_items[0], output_size = crop_size)

            transformed_items = [v2.functional.crop(item, i, j, h, w) for item in transformed_items]

        if self.hflip and torch.rand() > 0.5:
            transformed_items = [v2.functional.hflip(item) for item in transformed_items]

        return transformed_items
      

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (valid >= 0.5) & (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


        
if __name__=='__main__':
    import argparse


    parser = argparse.ArgumentParser(description='TMA')
    #training setting
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--lr', type=float, default=2e-4)


    #dataloader setting
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)

    #model setting
    parser.add_argument('--grad_clip', type=float, default=1)

    # loss setting
    parser.add_argument('--weight', type=float, default=0.8)

    #wandb setting
    parser.add_argument('--wandb', action='store_true', default=False)

    #Loading pretrained models
    parser.add_argument('--restore_ckpt_path', type=str, default="", help="Path to existing checkpoint to be loaded")
    parser.add_argument('--continue_training', action='store_true', default=False, help="Continue learning with previous params")
    
    #Data augmentation
    parser.add_argument('--augment', action="store_true", default=False, help = "Activate data augmentation : Random Cropping to (288,384), Horizontal Flip, Vertical Flip, and Spacial Transforms.")
    
    args = parser.parse_args()
    set_seed(1)
    if args.wandb:
        wandb_name = args.checkpoint_dir.split('/')[-1]
        wandb.init(name=wandb_name, project='TMA_DSEC_full')

    trainer = Trainer(args)
    trainer.train()
    
