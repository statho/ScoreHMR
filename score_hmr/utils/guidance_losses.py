import torch
from .geometry import perspective_projection


def gmof(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Geman-McClure error function.
    Args:
        x : Raw error signal
        sigma : Robustness hyperparameter
    Returns:
        torch.Tensor: Robust error signal
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def keypoint_fitting_loss(
    model_joints: torch.Tensor,
    camera_translation: torch.Tensor,
    joints_2d: torch.Tensor,
    joints_conf: torch.Tensor,
    camera_center: torch.Tensor,
    focal_length: torch.Tensor,
    img_size: torch.Tensor,
    sigma: float = 100.0,
) -> torch.Tensor:
    """
    Loss function for model fitting on 2D keypoints.
    Args:
        model_joints       (torch.Tensor) : Tensor of shape [B, NJ, 3] containing the SMPL 3D joint locations.
        camera_translation (torch.Tensor) : Tensor of shape [B, 3] containing the camera translation.
        joints_2d          (torch.Tensor) : Tensor of shape [B, N, 2] containing the target 2D joint locations.
        joints_conf        (torch.Tensor) : Tensor of shape [B, N, 1] containing the target 2D joint confidences.
        camera_center      (torch.Tensor) : Tensor of shape [B, 2] containing the camera center in pixels.
        focal_length       (torch.Tensor) : Tensor of shape [B, 2] containing focal length value in pixels.
        img_size           (torch.Tensor) : Tensor of shape [B, 2] containing the image size in pixels (height, width).
    Returns:
        torch.Tensor: Total loss value.
    """
    img_size = img_size.max(dim=-1)[0]

    # Heuristic for scaling data_weight with resolution used in SMPLify-X
    data_weight = (1000.0 / img_size).reshape(-1, 1, 1).repeat(1, 1, 2)

    # Project 3D model joints
    projected_joints = perspective_projection(
        model_joints, camera_translation, focal_length, camera_center=camera_center
    )

    # Compute robust reprojection loss
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (
        (data_weight**2) * (joints_conf**2) * reprojection_error
    ).sum(dim=(1, 2))

    return reprojection_loss


def multiview_loss(
    body_pose_6d: torch.Tensor, consistency_weight: float = 300.0
) -> torch.Tensor:
    """
    Loss function for multiple view refinement.
    Args:
        body_pose_6d : Tensor of shape (V, 23, 6) containing the 6D pose of V views of a person.
        consistency_weight : Pose consistency loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    mean_pose = body_pose_6d.mean(dim=0).unsqueeze(dim=0)
    pose_diff = ((body_pose_6d - mean_pose) ** 2).sum(dim=-1)
    consistency_loss = consistency_weight**2 * pose_diff.sum()
    total_loss = consistency_loss
    return total_loss


def smoothness_loss(pred_pose_6d: torch.Tensor) -> torch.Tensor:
    """
    Loss function for temporal smoothness.
    Args:
        pred_pose : Tensor of shape [N, 144] containing the 6D pose of N frames in a video.
    Returns:
        torch.Tensor : Total loss value.
    """
    pose_diff = ((pred_pose_6d[1:] - pred_pose_6d[:-1]) ** 2).sum(dim=-1)
    return pose_diff
