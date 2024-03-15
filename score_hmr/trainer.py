import os
from tqdm import tqdm
from typing import Dict
from yacs.config import CfgNode
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA
from constants import CHECKPOINT_DIR, RESULTS_DIR
from score_hmr.models import SMPL
from score_hmr.utils.train_utils import cycle, filter_based_on_pose
from score_hmr.utils import recursive_to, prepare_smpl_params, StandarizeImageFeatures
from score_hmr.utils.skeleton_renderer import SkeletonRenderer



class Trainer(object):
    def __init__(
        self,
        cfg: CfgNode,
        name: str,
        diffusion_model: torch.nn.Module,
        num_samples: int = 1,
        train_dataloader: torch.utils.data.DataLoader = None,
        val_dataloader: torch.utils.data.DataLoader = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.model = diffusion_model
        self.device = diffusion_model.device
        self.use_betas = diffusion_model.use_betas
        self.num_samples = num_samples
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.step = 0
        self.train_num_steps = self.cfg.TRAIN.TOTAL_STEPS

        # Set up optimizer.
        self.opt = Adam(diffusion_model.parameters(), lr=self.cfg.TRAIN.LR, betas=(0.9, 0.99))

        # Keep an EMA copy of the model weights.
        self.ema = EMA(diffusion_model, beta=0.995, update_every=10)

        # Set up directories for tensorboard and checkpoints.
        self.ckpt_folder = os.path.join(os.path.join(RESULTS_DIR, "checkpoints", name))
        tb_folder = os.path.join(os.path.join(RESULTS_DIR, "tensorboard", name))
        os.makedirs(self.ckpt_folder, exist_ok=True)
        os.makedirs(tb_folder, exist_ok=True)
        self.tb_writter = SummaryWriter(tb_folder)

        # Set up renderer for TB visualizations.
        self.renderer = SkeletonRenderer(self.cfg)

        # Set up class to standarize image features.
        self.img_feat_standarizer = StandarizeImageFeatures(
            backbone=self.cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
            use_betas=self.use_betas,
            device=self.device,
        )

        # Set up SMPL model.
        smpl_cfg = {k.lower(): v for k, v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg).to(self.device)

    def save(self, milestone: int) -> None:
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
        }
        torch.save(data, os.path.join(self.ckpt_folder, f"model-{milestone}.pt"))

    def load(self, milestone: int, name: str = 'score_hmr', use_default: bool = False) -> None:
        ckpt_dir = f"{CHECKPOINT_DIR}/{name}" if use_default else self.ckpt_folder
        data = torch.load(f"{ckpt_dir}/model-{milestone}.pt", map_location=self.device)
        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])

    def train(self) -> None:
        self.dl = cycle(self.train_dataloader)
        self.val_dl = cycle(self.val_dataloader)

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                batch = recursive_to(next(self.dl), self.device)

                # Keep entries with SMPL pose annotations.
                batch = filter_based_on_pose(batch)

                self.opt.zero_grad()

                # Compute training loss.
                simple_loss, _, _ = self.model(batch)
                if self.use_betas:
                    has_gt_pose = batch["has_smpl_params"]["body_pose"].bool()
                    has_gt_shape = batch["has_smpl_params"]["betas"].bool()
                    simple_loss_pose = simple_loss[:, :-10][has_gt_pose].reshape(-1)
                    simple_loss_shape = simple_loss[:, -10:][has_gt_shape].reshape(-1)
                    simple_loss = simple_loss_pose.mean() + simple_loss_shape.mean()
                else:
                    has_gt_pose = batch["has_smpl_params"]["body_pose"].bool()
                    simple_loss = simple_loss[has_gt_pose]
                    simple_loss = simple_loss.mean()
                loss = simple_loss

                # Optimizer step.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                pbar.set_description(f"loss: {loss.item():.4f}")
                self.step += 1
                self.ema.to(self.device)
                self.ema.update()

                if self.step != 0 and self.step % self.cfg.TRAIN.LOG_FREQ == 0:
                    output_train = {
                        "losses": {"p_loss": simple_loss.detach()}
                    }

                    ## Validation

                    # Get validation batch.
                    batch_val = recursive_to(next(self.val_dl), self.device)
                    batch_size_val = batch_val["img_feats"].size(0)
                    num_samples_val = self.cfg.TRAIN.NUM_VAL_SAMPLES
                    bs_times_samples_val = batch_size_val * num_samples_val

                    cond_feats_val = batch_val["img_feats"]
                    cond_feats_val = self.img_feat_standarizer(cond_feats_val)
                    cond_feats_val = (
                        cond_feats_val.view(batch_size_val, 1, -1)
                        .repeat(1, num_samples_val, 1)
                        .reshape(bs_times_samples_val, -1)
                    )

                    # Sample from the diffusion model.
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        sampling_output = self.ema.ema_model.sample(
                            batch_val,
                            cond_feats_val,
                            bs_times_samples_val,
                        )
                        x_0 = sampling_output['x_0']
                    self.ema.ema_model.train()

                    # Run the SMPL model.
                    pred_smpl_params = prepare_smpl_params(
                        x_0,
                        num_samples_val,
                        use_betas=self.use_betas,
                        betas_min = self.model.betas_min if self.use_betas else None,
                        betas_max = self.model.betas_max if self.use_betas else None,
                        pred_betas=batch_val["pred_betas"] if not self.use_betas else None,
                    )
                    pred_smpl_params["global_orient"] = pred_smpl_params["global_orient"].reshape(batch_size_val * num_samples_val, -1, 3, 3)
                    pred_smpl_params["body_pose"] = pred_smpl_params["body_pose"].reshape(batch_size_val * num_samples_val, -1, 3, 3)
                    pred_smpl_params["betas"] = pred_smpl_params["betas"].reshape(batch_size_val * num_samples_val, -1)
                    smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False )
                    pred_keypoints_3d = smpl_output.joints

                    pred_cam = batch_val["pred_cam"].unsqueeze(1).repeat(1, num_samples_val, 1)
                    focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size_val, num_samples_val, 2, device=self.device, dtype=pred_cam.dtype)
                    pred_cam_t = torch.stack(
                        [
                            pred_cam[:, :, 1],
                            pred_cam[:, :, 2],
                            2 * focal_length[:, :, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, :, 0] + 1e-9),
                        ],
                        dim=-1,
                    )

                    output_val = {
                        "pred_cam_t": pred_cam_t,
                        "pred_keypoints_3d": pred_keypoints_3d.reshape(batch_size_val, num_samples_val, -1, 3),
                    }

                    # Log to tensorboard.
                    milestone = self.step // self.cfg.TRAIN.LOG_FREQ
                    self.visuals_logging(batch, output_train, milestone, train=True)
                    self.visuals_logging(batch_val, output_val, milestone, train=False)

                    # Save model checkpoint.
                    if self.step != 0 and self.step % self.cfg.TRAIN.CHECKPOINT_FREQ == 0:
                        self.save(milestone)

                pbar.update(1)

    def visuals_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Function to perform tensorboard logging.
        Args:
            batch: Dictionary with the dataset labels.
            output: Dictionary with model predictions.
            step_count: training iteration iterations number.
            train: flag to indicate train or val mode.
        """
        if train:
            losses = output["losses"]
            for loss_name, val in losses.items():
                self.tb_writter.add_scalar("train/" + loss_name, val.detach().item(), step_count)
            return

        images = batch["img"]
        batch_size = images.size(0)
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        images = 255 * images.permute(0, 2, 3, 1).cpu().numpy()

        num_images = min(batch_size, 4)
        num_samples_per_image = min(self.cfg.TRAIN.NUM_VAL_SAMPLES, 8)

        gt_keypoints_2d = batch["keypoints_2d"]
        gt_keypoints_3d = batch["keypoints_3d"]

        pred_keypoints_3d = output["pred_keypoints_3d"]
        pred_cam_t = output["pred_cam_t"]

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        predictions = self.renderer(
            pred_keypoints_3d[:num_images, :num_samples_per_image],
            gt_keypoints_3d[:num_images],
            2 * gt_keypoints_2d[:num_images],
            images=images[:num_images],
            camera_translation=pred_cam_t[:num_images, :num_samples_per_image],
        )

        self.tb_writter.add_image("val/predictions", predictions.transpose((2, 0, 1)), step_count)
