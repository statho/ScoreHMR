import math
import numpy as np
import torch
from typing import Tuple, Optional, Dict
from .geometry import rot6d_to_rotmat


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cam_crop_to_full(
    cam_bbox: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length: float = 5000.0,
) -> torch.Tensor:
    """
    Compute perspective camera translation, given the weak-persepctive camera, the bounding box and the image dimensions.
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def normalize_betas(betas: torch.Tensor, betas_min: float, betas_max: float) -> torch.Tensor:
    """Normalize SMPL betas to [-1, 1]."""
    return 2 * (betas - betas_min) / (betas_max - betas_min) - 1


def unnormalize_betas(betas: torch.Tensor, betas_min: float, betas_max: float) -> torch.Tensor:
    """Get back the unnormalized SMPL betas."""
    return 0.5 * (betas_max - betas_min) * (betas + 1) + betas_min


def prepare_smpl_params(
        x: torch.Tensor,
        num_samples: int = 1,
        use_betas: bool = False,
        betas_min: Optional[torch.Tensor] = None,
        betas_max: Optional[torch.Tensor] = None,
        pred_betas: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    """
    Args:
        x : Tensor of shape [B*N, P] containing the SMPL paramaters x from the diffusion model (B: batch_size, N: num_samples, P: dimension of SMPL parameters).
        num_samples : Number of samples from the diffusion model.
        use_betas : Boolean indicating whether or not the diffusion model uses SMPL betas.
        pred_betas : Tensor of shape [B, 10] containing the SMPL betas from a regression network (not used when the DM models SMPL betas as well).
    Returns:
        Dict containing the SMPL model parameters.
    """
    batch_size = x.size(0) // num_samples
    if use_betas:
        pred_pose = x[:, :-10]
        # pred_betas = x[:, -10:]
        pred_betas = unnormalize_betas(x[:, -10:], betas_min, betas_max)
        pred_betas = pred_betas.reshape(batch_size, num_samples, -1)
    else:
        pred_pose = x
        pred_betas = pred_betas.unsqueeze(1).repeat(1, num_samples, 1)

    pred_pose = rot6d_to_rotmat(pred_pose.reshape(batch_size * num_samples, -1)).view(batch_size, num_samples, 24, 3, 3)
    pred_smpl_params = {
        "global_orient": pred_pose[:, 0, [0]] if num_samples == 1 else pred_pose[:, :, [0]],
        "body_pose": pred_pose[:, 0, 1:] if num_samples == 1 else pred_pose[:, :, 1:],
        "betas": pred_betas[:, 0] if num_samples == 1 else pred_betas,
    }
    return pred_smpl_params


class StandarizeImageFeatures:
    def __init__(
        self,
        backbone: str = "pare",
        use_betas: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[str] = 'tensor',
    ) -> None:
        feat_stats = np.load(f"data/stats/{backbone}_feat_stats.npz")
        if backbone == "prohmr":
            self.feat_mean = feat_stats["mean"]
            self.feat_std = feat_stats["std"]
        elif backbone == "pare":
            pose_feat_mean = feat_stats["pose_feats_mean"].reshape(-1)
            pose_feat_std = feat_stats["pose_feats_std"].reshape(-1)
            cam_shape_feat_mean = feat_stats["cam_shape_feats_mean"].reshape(-1)
            cam_shape_feat_std = feat_stats["cam_shape_feats_std"].reshape(-1)
            self.feat_mean = np.concatenate((pose_feat_mean, cam_shape_feat_mean)) if use_betas else pose_feat_mean
            self.feat_std = np.concatenate((pose_feat_std, cam_shape_feat_std)) if use_betas else pose_feat_std
        if dtype=='tensor':
            self.feat_mean = torch.from_numpy(self.feat_mean).to(device)
            self.feat_std = torch.from_numpy(self.feat_std).to(device)

    def __call__(self, feats) -> torch.Tensor:
        return (feats - self.feat_mean) / self.feat_std


class EarlyStopping:
    """
    Class to apply early stopping.
    It stores the early stopped samples in the batch in self.x_start_cached and (optionally) self.camera_translation_cached.
    """

    def __init__(
            self,
            shape: Tuple,
            device: torch.device,
            opt_trans: bool = False,
            tolerance: Optional[float] = 1e-5
        ) -> None:
        """
        Args:
            shape : shape of the diffusion model parameters.
            opt_trans: flag which indicates whether camera_translation should be optimized.
            tolerance: threshold hyperparameter.
        """
        self.prev_loss = None
        self.device = device
        self.opt_trans = opt_trans
        self.tol = tolerance
        self.x_start_cached = torch.zeros(shape, device=device)
        self.camera_translation_cached = torch.zeros((shape[0], 3), device=device) if opt_trans else None

    def get_stop_mask(self, loss: torch.Tensor) -> torch.Tensor:
        """ Returns a mask with the samples from the batch that coverge in the current step. """
        mask = torch.zeros_like(loss, device=self.device).bool()
        if self.prev_loss is None:
            self.prev_loss = loss.clone().detach()
        else:
            cur_loss = loss.clone().detach()
            rel_change = torch.abs(self.prev_loss - cur_loss) / torch.max(
                torch.max(self.prev_loss, cur_loss),
                torch.ones(cur_loss.shape[0], device=self.device),
            )
            self.prev_loss = cur_loss.clone()
            mask = rel_change < self.tol
        return mask

    def cache_results(self, mask: torch.Tensor, x_start: torch.Tensor, camera_translation: torch.Tensor) -> None:
        """ Cache samples that coverged in the current step. """
        if mask is not None and torch.any(mask).item():
            # Keep only the samples that have not converged yet.
            mask_valid = self.x_start_cached.sum(dim=-1) == 0
            mask_use = mask_valid & mask
            # Cache results that converged in the current step.
            self.x_start_cached[mask_use] = x_start[mask_use].clone().detach()
            if camera_translation is not None:
                self.camera_translation_cached[mask_use] = camera_translation[mask_use].clone().detach()

    def get_preds(self, x_start: torch.Tensor, camera_translation: torch.Tensor) -> Dict:
        """ Returns the final results. """
        mask_valid = self.x_start_cached.sum(dim=-1) == 0
        self.x_start_cached[mask_valid] = x_start[mask_valid].clone().detach()
        if camera_translation is not None:
            self.camera_translation_cached[mask_valid] = camera_translation[mask_valid].clone().detach()
        return {
            'x_0' : self.x_start_cached,
            'camera_translation': self.camera_translation_cached
        }