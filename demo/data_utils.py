"""
Code borrowed and adapted from:
https://github.com/vye16/slahmr/blob/main/slahmr/data/tools.py
"""
import os
import json
import functools
import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from constants import DEFAULT_MEAN, DEFAULT_STD
from score_hmr.datasets.utils import generate_image_patch, convert_cvimg_to_tensor


def read_keypoints(keypoint_fn):
    """
    Reads body keypoint data of a person.
    """
    empty_kps = np.zeros((44, 3), dtype=np.float32)
    if not os.path.isfile(keypoint_fn):
        return empty_kps

    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data["people"]) == 0:
        print("WARNING: Found no keypoints in %s! Returning zeros!" % (keypoint_fn))
        return empty_kps

    person_data = data["people"][0]
    body_keypoints = np.array(person_data["pose_keypoints_2d"], dtype=np.float32)
    body_keypoints = np.concatenate(
        (body_keypoints.reshape([-1, 3]), np.zeros((19, 3), dtype=np.float32)), axis=0
    )
    return body_keypoints


def read_smpl_cam_box(pred_path, num_betas=10, img_path=None):
    """
    Reads the SMPL parameters, camera and bbox of a smpl prediction exported from phalp outputs.
    """
    body_pose = np.zeros((23, 3), dtype=np.float32)
    global_orient = np.zeros(3, dtype=np.float32)
    betas = np.zeros(num_betas, dtype=np.float32)
    box = np.zeros(4, dtype=np.float32)
    box_center = np.zeros(2, dtype=np.float32)
    box_scale = np.zeros(2, dtype=np.float32)
    cam_bbox = np.zeros(3, dtype=np.float32)
    cam_trans = np.zeros(3, dtype=np.float32)

    if not os.path.isfile(pred_path):
        return body_pose, global_orient, betas, box, box_center, box_scale, cam_bbox, cam_trans

    with open(pred_path, "r") as f:
        data = json.load(f)

    if "body_pose" in data:
        body_pose = np.array(data["body_pose"], dtype=np.float32)
    if "global_orient" in data:
        global_orient = np.array(data["global_orient"], dtype=np.float32)
    if "betas" in data:
        betas = np.array(data["betas"], dtype=np.float32)
    if "cam_trans" in data:
        cam_trans = np.array(data["cam_trans"], dtype=np.float32)
    if "bbox" in data:
        box = np.array(data["bbox"], dtype=np.float32)
    if "box_center" in data:
        box_center = np.array(data["box_center"], dtype=np.float32)
    if "box_scale" in data:
        box_scale = np.array(data["box_scale"], dtype=np.float32)
    if "cam_bbox" in data:
        cam_bbox = np.array(data["cam_bbox"], dtype=np.float32)

    return body_pose, global_orient, betas, box, box_center, box_scale, cam_bbox, cam_trans


def load_smpl_cam_box(pred_paths, interp=True, num_betas=10):
    vis_mask = np.array([os.path.isfile(x) for x in pred_paths])
    vis_idcs = np.where(vis_mask)[0]

    # load monocular smpl predictions
    stack_fnc = functools.partial(np.stack, axis=0)
    body_pose, global_orient, betas, box, box_center, box_scale, cam_bbox, cam_trans = map(
        stack_fnc, zip(*[read_smpl_cam_box(p, num_betas=num_betas) for p in pred_paths])
    )
    if not interp:
        return {
            "body_pose": body_pose,
            "global_orient" : global_orient,
            "betas": betas,
            "box" : box,
            "box_center" : box_center,
            "box_scale" : box_scale,
            "cam_bbox" : cam_bbox,
            "cam_trans" : cam_trans,
        }

    # interpolate the occluded tracks
    orient_slerp = Slerp(vis_idcs, Rotation.from_rotvec(global_orient[vis_idcs]))
    cam_trans_interp = interp1d(vis_idcs, cam_trans[vis_idcs], axis=0)
    betas_interp = interp1d(vis_idcs, betas[vis_idcs], axis=0)
    cam_bbox_interp = interp1d(vis_idcs, cam_bbox[vis_idcs], axis=0)
    box_interp = interp1d(vis_idcs, box[vis_idcs], axis=0)
    box_center_interp = interp1d(vis_idcs, box_center[vis_idcs], axis=0)
    box_scale_interp = interp1d(vis_idcs, box_scale[vis_idcs], axis=0)

    tmin, tmax = min(vis_idcs), max(vis_idcs) + 1
    times = np.arange(tmin, tmax)

    # interpolate for each joint angle
    for i in range(body_pose.shape[1]):
        body_pose_slerp = Slerp(vis_idcs, Rotation.from_rotvec(body_pose[vis_idcs, i]))
        body_pose[times, i] = body_pose_slerp(times).as_rotvec()

    global_orient[times] = orient_slerp(times).as_rotvec()
    cam_trans[times] = cam_trans_interp(times)
    betas[times] = betas_interp(times)
    cam_bbox[times] = cam_bbox_interp(times)
    box[times] = box_interp(times)
    box_center[times] = box_center_interp(times)
    box_scale[times] = box_scale_interp(times)

    return {
        "body_pose": body_pose,
        "global_orient" : global_orient,
        "betas": betas,
        "box" : box,
        "box_center" : box_center,
        "box_scale" : box_scale,
        "cam_bbox" : cam_bbox,
        "cam_trans" : cam_trans,
    }


def read_images(img_path, box_center, box_scale, crop_res=224):
    if box_center.sum() > 0:
        # read images -- vis_mask 1 or 0
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError(f"Fail to read {img_path}")
        img_height, img_width, img_channels = cvimg.shape

        img_patch_cv, _ = generate_image_patch(
            img=cvimg,
            c_x=box_center[0],
            c_y=box_center[1],
            bb_width=box_scale.max(),
            bb_height=box_scale.max(),
            patch_width=crop_res,
            patch_height=crop_res,
            do_flip=False,
            scale=1.0,
            rot=0.0,
            load_image=True,
        )
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)
        for n_c in range(min(img_channels, 3)):
            img_patch[n_c, :, :] = (
                img_patch[n_c, :, :] - DEFAULT_MEAN[n_c]
            ) / DEFAULT_STD[n_c]
    else:
        # corresponds to vis_mask=-1 (track out of scene) and will not be used
        img_patch = np.zeros((3, crop_res, crop_res), dtype=np.float32)
    return img_patch
