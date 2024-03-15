"""
Code adapted from
https://github.com/nkolot/ProHMR/blob/master/prohmr/datasets/image_dataset.py
"""
import copy
import numpy as np
from os.path import join
from typing import Dict
from yacs.config import CfgNode
from .dataset import Dataset
from .utils import get_example
from score_hmr.utils import StandarizeImageFeatures
from constants import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD, FLIP_KEYPOINT_PERMUTATION


class ImageDataset(Dataset):

    def __init__(
        self,
        cfg: CfgNode,
        dataset_file: str,
        img_dir: str,
        train: bool = True,
        **kwargs
    ) -> None:
        """
        Dataset class used for loading images, corresponding annotations and optionally features/estimates from a regression network.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (disables loading images and uses the cached image features).
        """
        super(ImageDataset, self).__init__()
        self.cfg = cfg
        self.train = train
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.img_size = DEFAULT_IMG_SIZE
        self.img_dir = img_dir

        self.load_img_feats = self.cfg.EXTRA.LOAD_IMG_FEATS
        self.load_predicitons = self.cfg.EXTRA.LOAD_PREDICTIONS
        if train and self.load_img_feats:
            self.img_feat_standarizer = StandarizeImageFeatures(
                backbone=self.load_img_feats,
                use_betas=cfg.MODEL.USE_BETAS,
                dtype='numpy'
            )

        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        # Load the dataset.
        self.data = np.load(dataset_file)

        self.imgname = self.data["imgname"]

        # Bounding boxes are assumed to be in the center and scale format.
        self.center = self.data["center"].astype(np.float32)
        self.scale = self.data["scale"].astype(np.float32).reshape(len(self.center), -1).max(axis=-1) / 200


        # Get gt SMPL parameters, if available.
        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)
        try:
            self.body_pose = self.data["body_pose"].astype(np.float32)
            self.has_body_pose = self.data["has_body_pose"].astype(np.float32)
        except KeyError:
            self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data["betas"].astype(np.float32)
            self.has_betas = self.data["has_betas"].astype(np.float32)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # Get OpenPose 2d keypoints, if available.
        try:
            body_keypoints_2d = self.data["body_keypoints_2d"].astype(np.float32)
        except KeyError:
            body_keypoints_2d = np.zeros((len(self.center), 25, 3), dtype=np.float32)
        # Get GT 2d keypoints, if available.
        try:
            extra_keypoints_2d = self.data["extra_keypoints_2d"].astype(np.float32)
        except KeyError:
            extra_keypoints_2d = np.zeros((len(self.center), 19, 3), dtype=np.float32)
        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1)

        # Get GT 3d keypoints (OpenPose definition), if available.
        try:
            body_keypoints_3d = self.data["body_keypoints_3d"].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0
        # Get GT 3d keypoints, if available.
        try:
            extra_keypoints_3d = self.data["extra_keypoints_3d"].astype(np.float32)
        except KeyError:
            extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)
        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1)

        # Load cached image features.
        if self.load_img_feats == "prohmr":
            pred_file = np.load(kwargs["prohmr_preds_file"])
            self.img_feats = pred_file["img_feats"]
            if train:
                self.img_feats = self.img_feat_standarizer(self.img_feats)
        if self.load_img_feats  == "pare":
            pred_file = np.load(kwargs["pare_preds_file"])
            self.img_feats = pred_file["pose_feats"].reshape(pred_file["pose_feats"].shape[0], -1)
            if cfg.MODEL.USE_BETAS:
                cam_shape_feats = pred_file["cam_shape_feats"].reshape(pred_file['cam_shape_feats'].shape[0], -1)
                self.img_feats = np.concatenate((self.img_feats, cam_shape_feats), axis=1)
            if train:
                self.img_feats = self.img_feat_standarizer(self.img_feats)

        # Load cached predictions from regresssion.
        if self.load_predicitons == "prohmr":
            pred_file = np.load(kwargs["prohmr_preds_file"])
        elif self.load_predicitons == "pare":
            pred_file = np.load(kwargs["pare_preds_file"])
        elif self.load_predicitons == "hmr2":
            pred_file = np.load(kwargs["hmr2_preds_file"])
        if self.load_predicitons:
            self.pred_pose = pred_file["pred_pose"] # N, 24, 3, 3
            self.pred_betas = pred_file["pred_betas"]
            self.pred_cam = pred_file["pred_cam"]
            self.pred_cam_t = pred_file["pred_cam_t"] if "pred_cam_t" in pred_file.__dict__['files'] else None

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file = join(self.img_dir, self.imgname[idx].decode("utf-8"))
        except AttributeError:
            image_file = join(self.img_dir, self.imgname[idx])
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx] * 200

        body_pose = self.body_pose[idx].copy()
        betas = self.betas[idx].copy()
        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        smpl_params = {
            "global_orient": body_pose[:3],
            "body_pose": body_pose[3:],
            "betas": betas,
        }
        has_smpl_params = {
            "global_orient": has_body_pose,
            "body_pose": has_body_pose,
            "betas": has_betas,
        }
        smpl_params_is_axis_angle = {
            "global_orient": True,
            "body_pose": True,
            "betas": False,
        }

        augm_config = self.cfg.DATASETS.CONFIG

        # Crop image and (possibly) perform data augmentation.
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            image_file,
            center_x,
            center_y,
            bbox_size,
            bbox_size,
            keypoints_2d,
            keypoints_3d,
            smpl_params,
            has_smpl_params,
            flip_kp_permutation=self.flip_keypoint_permutation,
            patch_width=self.img_size,
            patch_height=self.img_size,
            mean=self.mean,
            std=self.std,
            do_augment=False,
            augm_config=augm_config,
            load_image=not self.train,
        )

        item = {}
        # These are the keypoints in the original image coordinates (before cropping).
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item["img"] = img_patch if img_patch is not None else np.zeros((3, 4, 4), dtype=np.float32)
        item["orig_keypoints_2d"] = orig_keypoints_2d
        item["keypoints_2d"] = keypoints_2d
        item["keypoints_3d"] = keypoints_3d
        item["box_center"] = self.center[idx].copy()
        item["box_size"] = self.scale[idx] * 200
        item["img_size"] = 1.0 * img_size[::-1].copy() if img_size is not None else np.zeros((2), dtype=np.float32)
        item["smpl_params"] = smpl_params
        item["has_smpl_params"] = has_smpl_params
        item["smpl_params_is_axis_angle"] = smpl_params_is_axis_angle
        item["imgname"] = image_file
        item["idx"] = idx

        if self.load_img_feats is not None:
            item["img_feats"] = self.img_feats[idx]
        if self.load_predicitons is not None:
            item["pred_pose"] = self.pred_pose[idx]
            item["pred_betas"] = self.pred_betas[idx]
            item["pred_cam"] = self.pred_cam[idx]
            if self.pred_cam_t is not None:
                item["pred_cam_t"] = self.pred_cam_t[idx]

        return item