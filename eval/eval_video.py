"""
Evaluate ScoreHMR on temporal model fitting.

Example:
python eval/eval_video.py --dataset 3DPW-TEST-VIDEO --use_default_ckpt

Running the above will compute the MPJPE, Reconstruction Error and Acceleration Error
before and after temporal model fitting for the test set of 3DPW.
The code uses cached HMR 2.0b predictions.
"""

import torch
torch.manual_seed(0)
import argparse
from tqdm import tqdm

from score_hmr.utils import *
from score_hmr.configs import dataset_config, model_config
from score_hmr.datasets import create_dataset
from score_hmr.models.model_utils import load_diffusion_model, load_pare


NUM_SAMPLES = 1


parser = argparse.ArgumentParser(description="Evaluate ScoreHMR on temporal model fitting.")
parser.add_argument("--name", type=str, default='score_hmr', help="Name of experiment.")
parser.add_argument("--dataset", type=str, default="3DPW-TEST-VIDEO", help="Dataset to evaluate.")
parser.add_argument("--predictions", type=str, default="hmr2", help="Predictions from regression model.")
parser.add_argument("--milestone", type=int, default=100, help="Milestone to use for evaluation.")
parser.add_argument("--use_default_ckpt", action='store_true', default=False, help="Use weights from pretrained models provided in the repo.")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load config.
model_cfg = model_config()
model_cfg.defrost()
model_cfg.EXTRA.LOAD_PREDICTIONS = args.predictions
model_cfg.freeze()

# Dataset.
dataset_cfg = dataset_config()[args.dataset]
dataset = create_dataset(model_cfg, dataset_cfg, train=False)

# Load PARE model.
pare = load_pare(model_cfg.SMPL).to(device)
pare.eval()

img_feat_standarizer = StandarizeImageFeatures(
    backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
    use_betas=False,
    device=device,
)

# Load diffusion model.
extra_args = {
    "name" : args.name,
    "keypoint_guidance": True,
    "temporal_guidance": True,
    "use_default_ckpt": args.use_default_ckpt,
    "device": device,
}
diffusion_model = load_diffusion_model(model_cfg, **extra_args)

# Set up Evaluator.
output = None
reg_accel_errors, opt_accel_errors = [], []
metrics = ["reg_mpjpe", "reg_re", "opt_mpjpe", "opt_re"]
evaluator = Evaluator(dataset_length=dataset.total_length(), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)


### Eval Loop ###

for ii, batch in enumerate(tqdm(dataset)):
    batch = recursive_to(batch, device)
    batch_size = batch["keypoints_3d"].size(0)

    # Get 3D keypoints from regression for evaluation.
    output = {}
    batch["pred_betas"] = batch["pred_betas"].mean(dim=0).unsqueeze(dim=0).repeat(batch_size, 1) # avg pool betas
    reg_smpl_params = {
        "betas": batch["pred_betas"],
        "global_orient": batch["pred_pose"][:, [0]],
        "body_pose": batch["pred_pose"][:, 1:],
    }
    smpl_out_reg = diffusion_model.smpl(**reg_smpl_params, pose2rot=False)
    output["pred_keypoints_3d"] = smpl_out_reg.joints.unsqueeze(1)

    # Get image features.
    with torch.no_grad():
        pare_out = pare(batch["img"], get_feats=True)
    cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
    cond_feats = img_feat_standarizer(cond_feats) # normalize image features

    # Prepare things for model fitting.
    batch["camera_center"] = 0.5 * batch["img_size"]
    gt_keypoints_2d = batch["orig_keypoints_2d"].clone()
    batch["joints_2d"] = gt_keypoints_2d[:, :, :2]
    batch["joints_conf"] = gt_keypoints_2d[:, :, [-1]]
    batch["focal_length"] = model_cfg.EXTRA.FOCAL_LENGTH * torch.ones_like(batch["camera_center"])
    batch["init_cam_t"] = batch["pred_cam_t"]


    # Run iterative refinement with ScoreHMR.
    with torch.no_grad():
        dm_out = diffusion_model.sample(
            batch, cond_feats, batch_size=batch_size * NUM_SAMPLES
        )

    # Get 3D keypoints after multi-view refinemt with ScoreHMR.
    opt_output = {}
    pred_smpl_params = prepare_smpl_params(
        dm_out['x_0'],
        num_samples = NUM_SAMPLES,
        use_betas = False,
        pred_betas=batch["pred_betas"],
    )
    smpl_out = diffusion_model.smpl(**pred_smpl_params, pose2rot=False)
    opt_output["pred_keypoints_3d"] = smpl_out.joints.unsqueeze(1)

    # Evaluation
    evaluator(output, batch, opt_output)
    evaluator.log()

    # Normalize 3D keypoints with the pelvis position.
    gt_keypoints_3d = (
        batch["keypoints_3d"][:, :, :-1]
        - batch["keypoints_3d"][:, [model_cfg.EXTRA.PELVIS_IND], :-1]
    )
    pred_keypoints_3d = (
        output["pred_keypoints_3d"][:, 0]
        - output["pred_keypoints_3d"][:, 0, [model_cfg.EXTRA.PELVIS_IND]]
    )
    opt_keypoints_3d = (
        opt_output["pred_keypoints_3d"][:, 0]
        - opt_output["pred_keypoints_3d"][:, 0, [model_cfg.EXTRA.PELVIS_IND]]
    )
    # Compute acceleration error.
    reg_accel_error = (
        1000
        * compute_error_accel(
            gt_keypoints_3d[:, dataset_cfg.KEYPOINT_LIST],
            pred_keypoints_3d[:, dataset_cfg.KEYPOINT_LIST],
        ).cpu().numpy()
    )
    opt_accel_error = (
        1000
        * compute_error_accel(
            gt_keypoints_3d[:, dataset_cfg.KEYPOINT_LIST],
            opt_keypoints_3d[:, dataset_cfg.KEYPOINT_LIST],
        ).cpu().numpy()
    )
    reg_accel_errors.append(reg_accel_error)
    opt_accel_errors.append(opt_accel_error)
    print("reg_accel: {:.1f}".format(np.concatenate(reg_accel_errors).mean()))
    print("opt_accel:  {:.1f}".format(np.concatenate(opt_accel_errors).mean()))

print("Finished evaluation!")
evaluator.log()
print("reg_accel: {:.1f}".format(np.concatenate(reg_accel_errors).mean()))
print("opt_accel:  {:.1f}".format(np.concatenate(opt_accel_errors).mean()))