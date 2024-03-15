"""
Code adapted from
https://github.com/nkolot/ProHMR/blob/master/prohmr/utils/pose_utils.py
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3)
        joints_pred (Nx14x3)
        vis (N)
    Returns:
        error_accel (N-2)
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = torch.norm(accel_pred - accel_gt, dim=2)
    if vis is None:
        new_vis = torch.ones(len(normed), dtype=torch.bool)
    else:
        invis = ~vis
        invis1 = torch.roll(invis, -1)
        invis2 = torch.roll(invis, -2)
        new_invis = invis | invis1 | invis2
        new_invis = new_invis[:-2]
        new_vis = ~new_invis
    return torch.mean(normed[new_vis], dim=1)


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1, 2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale * torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)


def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt(((S1_hat - S2) ** 2).sum(dim=-1)).mean(dim=-1)
    return re.cpu().numpy()


def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = (
        torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1))
        .mean(dim=-1)
        .cpu()
        .numpy()
    )

    # Reconstuction_error
    r_error = reconstruction_error(pred_joints.cpu(), gt_joints.cpu())
    return 1000 * mpjpe, 1000 * r_error


class Evaluator:
    def __init__(
        self,
        dataset_length: int,
        keypoint_list: List,
        pelvis_ind: int,
        metrics: List = ["reg_mpjpe", "reg_re"],
    ):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind     (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics       [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.keypoint_list = keypoint_list
        self.pelvis_ind = pelvis_ind
        self.metrics = metrics
        for metric in self.metrics:
            setattr(self, metric, np.zeros((dataset_length,)))
        self.counter = 0

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print("Evaluation has not started")
            return
        print(f"{self.counter} / {self.dataset_length} samples")
        for metric in self.metrics:
            print(f"{metric}: {getattr(self, metric)[:self.counter].mean():.1f} mm")
        print("***")

    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None):
        """
        Evaluate current batch.
        Args:
            output     (Dict): Regression output.
            batch      (Dict): Dictionary containing images and their corresponding annotations.
            opt_output (Dict): Optimization output.
        """

        if output is not None:
            pred_keypoints_3d = output["pred_keypoints_3d"]
            batch_size = pred_keypoints_3d.shape[0]
            num_samples = pred_keypoints_3d.shape[1]
            pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        else:
            batch_size = opt_output["pred_keypoints_3d"].shape[0]
            num_samples = 1

        # Normalize ground-truth 3D keypoints.
        gt_keypoints_3d = (
            batch["keypoints_3d"][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
        )
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]  # B, N, 44, 3

        # Get regression errors.
        if output is not None:
            mpjpe, re = eval_pose(
                pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[:, self.keypoint_list],
                gt_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[:, self.keypoint_list]
            )
            mpjpe = mpjpe.reshape(batch_size, num_samples)
            re = re.reshape(batch_size, num_samples)

        # Get optimization errors.
        if opt_output is not None:
            opt_keypoints_3d = opt_output["pred_keypoints_3d"]
            opt_keypoints_3d -= opt_keypoints_3d[:, :, [self.pelvis_ind]]
            opt_mpjpe, opt_re = eval_pose(
                opt_keypoints_3d[:, 0, self.keypoint_list],
                gt_keypoints_3d[:, 0, self.keypoint_list],
            )

        ### Store results ###

        if hasattr(self, "reg_mpjpe"):
            reg_mpjpe = mpjpe[:, 0]
            self.reg_mpjpe[self.counter : self.counter + batch_size] = reg_mpjpe
        if hasattr(self, "reg_re"):
            reg_re = re[:, 0]
            self.reg_re[self.counter : self.counter + batch_size] = reg_re
        if hasattr(self, "opt_mpjpe"):
            self.opt_mpjpe[self.counter : self.counter + batch_size] = opt_mpjpe
        if hasattr(self, "opt_re"):
            self.opt_re[self.counter : self.counter + batch_size] = opt_re
        # Mean, std, min
        if hasattr(self, "min_mpjpe"):
            min_mpjpe = mpjpe.min(axis=-1)
            self.min_mpjpe[self.counter : self.counter + batch_size] = min_mpjpe
        if hasattr(self, "min_re"):
            min_re = re.min(axis=-1)
            self.min_re[self.counter : self.counter + batch_size] = min_re
        if hasattr(self, "mean_mpjpe"):
            mean_mpjpe = mpjpe.mean(axis=-1)
            self.mean_mpjpe[self.counter : self.counter + batch_size] = mean_mpjpe
        if hasattr(self, "std_mpjpe"):
            std_mpjpe = mpjpe.std(axis=-1)
            self.std_mpjpe[self.counter : self.counter + batch_size] = std_mpjpe
        if hasattr(self, "mean_re"):
            mean_re = re.mean(axis=-1)
            self.mean_re[self.counter : self.counter + batch_size] = mean_re
        if hasattr(self, "std_re"):
            std_re = re.std(axis=-1)
            self.std_re[self.counter : self.counter + batch_size] = std_re

        self.counter += batch_size
