import os
import torch
import argparse
import numpy as np
from score_hmr.utils import recursive_to
from score_hmr.datasets import create_dataset
from score_hmr.configs import dataset_config, model_config
from score_hmr.models.model_utils import load_pare


# Training and Validation datasets.
DATASETS = ["H36M-TRAIN", "COCO-TRAIN", "MPI-INF-TRAIN", "H36M-VAL-P2", "COCO-VAL"]
OUT_DIR = 'cache/pare'
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for data loading.")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_cfg = model_config()
dataset_cfg = dataset_config()

# Load PARE model.
pare = load_pare(model_cfg.SMPL).to(device)
pare.eval()


for dset in DATASETS:
    betas_ = []
    pose_ = []
    pred_cam_ = []
    pose_feats_ = []
    cam_shape_feats_ = []

    print(f"Generating predictions on {dset}")
    npz_path = f"{OUT_DIR}/{os.path.basename(dataset_cfg[dset]['DATASET_FILE'])}"

    dataset = create_dataset(model_cfg, dataset_cfg[dset], train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            pare_out = pare(batch["img"])

        betas_.append(pare_out['pred_shape'].cpu().numpy())
        pose_.append(pare_out['pred_pose'].cpu().numpy())
        pred_cam_.append(pare_out['pred_cam'].cpu().numpy())
        pose_feats_.append(pare_out['pose_feats'].cpu().numpy())
        cam_shape_feats_.append(pare_out['cam_shape_feats'].cpu().numpy())

    print('Saving npz file')
    np.savez(
        npz_path,
        pose_feats = np.concatenate(pose_feats_),
        cam_shape_feats = np.concatenate(cam_shape_feats_),
        pred_betas = np.concatenate(betas_),
        pred_pose = np.concatenate(pose_),
        pred_cam = np.concatenate(pred_cam_),
    )
