"""
Evaluate ScoreHMR on single-frame model fitting.

Example usage:
python eval/eval_keypoint_fitting.py --dataset 3DPW-TEST --shuffle --use_default_ckpt

Running the above will compute the MPJPE and Reconstruction Error before and after fitting for the test set of 3DPW.
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


parser = argparse.ArgumentParser(description="Evaluate ScoreHMR on single-frame model fitting.")
parser.add_argument("--name", type=str, default='score_hmr', help="Name of experiment.")
parser.add_argument("--dataset", type=str, default="3DPW-TEST", help="Dataset to evaluate.")
parser.add_argument("--predictions", type=str, default="hmr2", help="Predictions from regression model." )
# Dataloader
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference.")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers used for data loading.")
parser.add_argument("--log_freq", type=int, default=10, help="How often to log results.")
parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the dataset during evaluation.")
# Model
parser.add_argument("--use_betas", action="store_true", default=False, help="If true, uses SMPL shape parameters in the diffusion model.")
parser.add_argument("--milestone", type=int, default=100, help="Milestone to use for evaluation.")
parser.add_argument("--use_default_ckpt", action='store_true', default=False, help="Use weights from pretrained models provided in the repo.")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Load config.
model_cfg = model_config()
model_cfg.defrost()
model_cfg.MODEL.USE_BETAS = args.use_betas
model_cfg.EXTRA.LOAD_PREDICTIONS = args.predictions
model_cfg.freeze()

# Dataset.
dataset_cfg = dataset_config()[args.dataset]
dataset = create_dataset(model_cfg, dataset_cfg, train=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    args.batch_size,
    shuffle=args.shuffle,
    drop_last=False,
    num_workers=args.num_workers,
)

# Load PARE model.
pare = load_pare(model_cfg.SMPL).to(device)
pare.eval()

img_feat_standarizer = StandarizeImageFeatures(
    backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
    use_betas=args.use_betas,
    device=device,
)

# Load diffusion model.
extra_args = {
    "name" : args.name,
    "keypoint_guidance": True,
    "early_stopping": True,
    "use_default_ckpt": args.use_default_ckpt,
    "device": device,
}
diffusion_model = load_diffusion_model(model_cfg, **extra_args)

# Set up Evaluator.
output = None
metrics = ["reg_mpjpe", "reg_re", "opt_mpjpe", "opt_re"]
evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)


### Eval Loop ###

for ii, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    batch_size = batch["keypoints_3d"].size(0)

    # Get 3D keypoints from regression for evaluation.
    output = {}
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
    cond_feats = (
                    pare_out["pose_feats"].reshape(batch_size, -1)
                    if not args.use_betas
                    else torch.cat((pare_out["pose_feats"], pare_out["cam_shape_feats"]), dim=1).reshape(batch_size, -1)
    )
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

    # Get 3D keypoints after model fitting with ScoreHMR.
    opt_output = {}
    pred_smpl_params = prepare_smpl_params(
        dm_out['x_0'],
        num_samples = NUM_SAMPLES,
        use_betas = args.use_betas,
        betas_min = diffusion_model.betas_min if args.use_betas else None,
        betas_max = diffusion_model.betas_max if args.use_betas else None,
        pred_betas=batch["pred_betas"] if not args.use_betas else None,
    )
    smpl_out = diffusion_model.smpl(**pred_smpl_params, pose2rot=False)
    opt_output["pred_keypoints_3d"] = smpl_out.joints.unsqueeze(1)

    # Evaluation
    evaluator(output, batch, opt_output)
    if (ii + 1) % args.log_freq == 0:
        evaluator.log()

print("Evaluation finished!")
evaluator.log()