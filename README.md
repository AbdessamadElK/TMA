# Setup
1. Create an anaconda environment: `conda create -n TMA`.
2. Activate the invironment and install **Pytorch** according to your setup
3. Install the other requirements from "requirements.txt"
4. Download ffmpeg and add it to your Path environment variable
5. Import imageio and run the following command in interactive mode: `imageio.plugins.freeimage.downlaod()`

# Download and generate the DSEC Dataset
## Download the original dataset
Download the full DSEC dataset from here: [https://dsec.ifi.uzh.ch/dsec-datasets/download/] <br>

**The dataset folder must include:**
* train_events
* train_optical flow
* test_events
* test_forward_optical_flow_timestamps<br>

**Optional data:**
* train_images
* test_images
* train_images_distorted (images remaped towards events)
* test_images_distorted
* train_segmentation
* test_segmentation

You can download the remaped (distorted) images from here : [https://dsec.ifi.uzh.ch/dsec-detection/]

## Generate the new version of the dataset
For faster training, we use a different version of the dataset where events are saved as voxel grids with 15 bins in `.npz` files, and the corresponding optical flows are saved in `.npy` files. Therefore, after downloading the dataset, it must be transformed to the new format using the following command while in the TMA folder: <br>

`python ./data_utils/gen_dsec.py --dsec <Path_to_DSEC> --split <[train]/test/all>` <br>

**Other Options**<br>
* `-i` or `--images` : Include images <br>
* `-d` or `--distorted` : Use distorted images (towards events) instead of rectified ones <br>
* `--segmentation <Path_to_segmentation>` : Also include semantic segmentation data from *<Path_to_segmentation>* <br>

The segmentation folder must include generated `.png` files of images that have a corresponding optical flow, following the naming convention of the generated dataset: ("000000.png" - "000001.png" - ...).<br>

# Train the network
## Full training
To launch a full training, run the following command :<br>

`python train.py --checkpoint_dir "./ckpts/<label>" --num_steps <number of steps> --lr <learning_rate> --wandb`

## Train with validation
To train with validation, you must split the dataset from "TMA/datasets/dsec_full/trainval" into "TMA/datasets/dsec_split/train" and "TMA/datasets/dsec_split/val". You can follow the split example that is specified in the `.txt` files in the target directories.<br>

Then use the following command:<br>

`python train_split.py --checkpoint_dir "./ckpts/<label>" --num_steps <number of steps> --lr <learning_rate> --wandb`


