import os
import torch
import argparse
import shutil
import numpy as np
from hmr2.datasets import create_dataset
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.renderer import cam_crop_to_full
from score_hmr.utils import recursive_to
from score_hmr.configs import dataset_config


DATASETS = ["3DPW-TEST"]
OUT_DIR = 'cache/hmr2b'
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for data loading.")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset_cfg = dataset_config()


# Load HMR 2.0b.
download_models(CACHE_DIR_4DHUMANS)
# Copy SMPL model to the appropriate path for HMR 2.0 if it does not exist.
if not os.path.isfile(f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl'):
    shutil.copy('data/smpl/SMPL_NEUTRAL.pkl', f'{CACHE_DIR_4DHUMANS}/data/smpl/')
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
model = model.to(device)
model.eval()


for dset in DATASETS:
    betas_ = []
    pose_ = []
    pred_cam_ = []
    pred_cam_t_ = []

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
            out = model(batch)

        pred_cam_t_full = cam_crop_to_full(
            out['pred_cam'],
            batch['box_center'].float(),
            batch['box_size'].float(),
            batch['img_size'].float(),
        ).cpu().numpy()

        betas_.append( out['pred_smpl_params']['betas'].cpu().numpy() )
        pose_.append( torch.cat( (out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose'] ), axis=1).cpu().numpy() )
        pred_cam_.append( out['pred_cam'].cpu().numpy() )
        pred_cam_t_.append( pred_cam_t_full )

    print('Saving npz file')
    np.savez(
        npz_path,
        pred_betas = np.concatenate(betas_),
        pred_pose = np.concatenate(pose_),
        pred_cam = np.concatenate(pred_cam_),
        pred_cam_t = np.concatenate(pred_cam_t_),
    )
