"""
Script for training ScoreHMR.
Example usage:
python train.py --name <name_of_experiment>

Running the above command will use the default config file to train ScoreHMR, conditioned on PARE image features.
"""

import torch
torch.manual_seed(0)
import argparse
from score_hmr.datasets import MixedDataset
from score_hmr.configs import dataset_config, model_config
from score_hmr import GaussianDiffusion, Trainer
from score_hmr.models import FC

parser = argparse.ArgumentParser(description="Diffusion model training code.")
parser.add_argument("--name", type=str, required=True, help="Name of experiment.")
parser.add_argument("--use_betas", action="store_true", default=False, help="Indicate whether on not to use the SMPL shape parameters in the diffusion model.")
parser.add_argument("--img_feats", type=str, default="pare", help="Backbone to use image features from.")
parser.add_argument("--predictions", type=str, default="pare", help="Predictions from regression model (used only for logging purposes here)." )
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load configs.
model_cfg = model_config()
model_cfg.defrost()
model_cfg.MODEL.USE_BETAS = args.use_betas
model_cfg.EXTRA.LOAD_IMG_FEATS = args.img_feats
model_cfg.EXTRA.LOAD_PREDICTIONS = args.predictions
model_cfg.freeze()

# Set up datasets.
dataset_cfg = dataset_config()
train_dataset = MixedDataset(model_cfg, dataset_cfg, train=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=model_cfg.TRAIN.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=model_cfg.TRAIN.NUM_WORKERS,
)

val_dataset = MixedDataset(model_cfg, dataset_cfg, train=False)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=model_cfg.TRAIN.BATCH_SIZE_VAL,
    shuffle=True,
    drop_last=True,
    num_workers=model_cfg.TRAIN.NUM_WORKERS,
)

# Set up denoining model.
model = FC(model_cfg).to(device)

# Set up diffusion model.
diffusion = GaussianDiffusion(model_cfg, model, device=device).to(device)

# Set up Trainer and train the diffusion model.
trainer = Trainer(model_cfg, args.name, diffusion, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
trainer.train()