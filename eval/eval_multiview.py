"""
Evaluate ScoreHMR on multi-view refinement.

Example:
python eval/eval_multiview.py --dataset H36M-MULTIVIEW --use_default_ckpt

Running the above will compute the MPJPE and Reconstruction Error before and after multi-view refinemente for the test set of Human3.6M.
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


parser = argparse.ArgumentParser(description="Evaluate ScoreHMR on multi-view refinement.")
parser.add_argument("--name", type=str, default='score_hmr', help="Name of experiment.")
parser.add_argument("--dataset", type=str, default="H36M-MULTIVIEW", help="Dataset to evaluate.", choices=['H36M-MULTIVIEW', 'MANNEQUIN'])
parser.add_argument("--predictions", type=str, default="hmr2", help="Predictions from regression model." )
parser.add_argument("--log_freq", type=int, default=100, help="How often to log results.")
parser.add_argument("--milestone", type=int, default=100, help="Milestone to use for evaluation.")
parser.add_argument("--use_default_ckpt", action='store_true', default=False, help="Use weights from pretrained models provided in the repo.")
parser.add_argument("--ddim_step_size", type=int, default=10, help="DDIM step size.")
parser.add_argument("--optim_iters", type=int, default=2, help="Total optimization iterations.")
parser.add_argument("--sample_start", type=int, default=100, help="Timestep as a starting point for score guidance.")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load config.
model_cfg = model_config()
model_cfg.defrost()
model_cfg.EXTRA.LOAD_PREDICTIONS = args.predictions
model_cfg.GUIDANCE.OPTIM_ITERS = args.optim_iters
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
    "multiview_guidance": True,
    "use_default_ckpt": args.use_default_ckpt,
    "optim_iters" : args.optim_iters,
    "sample_start": args.sample_start,
    "ddim_step_size": args.ddim_step_size,
    "device": device,
}
diffusion_model = load_diffusion_model(model_cfg, **extra_args)


# Set up Evaluator.
output = None
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
    smpl_out = diffusion_model.smpl(**reg_smpl_params, pose2rot=False)
    output["pred_keypoints_3d"] = smpl_out.joints.unsqueeze(1)

    # Get image features.
    with torch.no_grad():
        pare_out = pare(batch["img"], get_feats=True)
    cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
    cond_feats = img_feat_standarizer(cond_feats) # normalize image features


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
    if (ii + 1) % args.log_freq == 0:
        evaluator.log()

print("Evaluation finished!")
evaluator.log()