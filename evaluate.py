from tqdm import tqdm
import numpy as np
import torch
import os
from dataloader.dsec_split import DSECsplit

import flow_vis
from model.TMA import TMA


@torch.no_grad()                   
def validate_DSEC(model):
    model.eval()
    val_dataset = DSECsplit('val')
    
    epe_list = []
    out_list = []

    # Visualization index
    vis_idx = np.random.randint(0, len(val_dataset)) 
    random_vis = None   

    bar = tqdm(enumerate(val_dataset),total=len(val_dataset), ncols=60)
    bar.set_description('Test')
    for index, (voxel1, voxel2, flow_map, valid2D) in bar:
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda() 
        flow_pred = model(voxel1, voxel2)[0].cpu()#[1,2,H,W]

        epe = torch.sum((flow_pred- flow_map)**2, dim=0).sqrt()#[H,W]
        mag = torch.sum(flow_map**2, dim=0).sqrt()#[H,W]

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid2D.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # Generate visualization
        if index == vis_idx:
            random_vis = flow_vis.flow_to_color(flow_pred.numpy().transpose(1, 2, 0), convert_to_bgr = False)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    
    print("Validation DSEC-TEST: %f, %f" % (epe, f1))
    return {'dsec-epe': epe, 'dsec-f1': f1, 'visualization': random_vis}


if __name__ == "__main__":
   
    from argparse import ArgumentParser
    

    parser =ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to a saved checkpoint file (.pth)")
    parser.add_argument("-b", "--input_bins", type=int, default=15, help="Number of input bins")

    args = parser.parse_args()

    # Load model
    model = TMA(input_bins=args.input_bins)
    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.cuda()

    # Run validation
    results = validate_DSEC(model)
    print("Validation EPE :", results['dsec-epe'])
    print("Validation outlier :", results['dsec-f1'])

    # # Save visualization
    # from datetime import datetime
    # from pathlib import Path
    # import imageio as io

    # filename = f"vis_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"
    # savepath = Path("./validation_vis")
    # savepath.mkdir(parents=True, exist_ok=True)
    # savepath = savepath / filename

    # vis = results['visualization']
    # io.imwrite(str(savepath), vis)


