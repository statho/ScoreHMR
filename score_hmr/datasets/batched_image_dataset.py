"""
Code adapted from
https://github.com/nkolot/ProHMR/blob/master/prohmr/datasets/batched_image_dataset.py
"""
import os
import copy
import numpy as np
import pickle
import torch
from yacs.config import CfgNode
from .dataset import Dataset
from .utils import get_example
from constants import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD, FLIP_KEYPOINT_PERMUTATION


class BatchedImageDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        dataset_file: str,
        img_dir: str,
        train: bool = False,
        **kwargs
    ) -> None:
        """
        Batched version of ImageDataset, where instead of a single example a list of examples is loaded (e.g. multiple views, all frames in a video).
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not.
        """
        super(BatchedImageDataset, self).__init__()
        self.cfg = cfg
        self.train = train
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.img_size = DEFAULT_IMG_SIZE
        self.img_dir = img_dir
        self.load_predicitons = self.cfg.EXTRA.LOAD_PREDICTIONS

        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        # Load the dataset.
        self.data = pickle.load(open(dataset_file, "rb"))

        # Load cached predictions.
        if self.load_predicitons=='hmr2':
            self.predictions = pickle.load(open(kwargs["hmr2_preds"], "rb"))

    def __len__(self) -> int:
        return len(self.data)

    def total_length(self) -> int:
        """
        Return the total number of images in the dataset.
        """
        return sum([len(datum["imgname"]) for datum in self.data])

    def __getitem__(self, idx: int):
        data = self.data[idx]
        num_images = len(data["imgname"])
        augm_config = self.cfg.DATASETS.CONFIG
        img_patch = []
        keypoints_2d = []
        keypoints_3d = []
        smpl_params = []
        has_smpl_params = []
        smpl_params_is_axis_angle = []
        img_size = []
        imgnames = []

        try:
            body_keypoints_2d = data["body_keypoints_2d"].astype(np.float32)
        except KeyError:
            body_keypoints_2d = np.zeros((num_images, 25, 3), dtype=np.float32)
        try:
            extra_keypoints_2d = data["extra_keypoints_2d"].astype(np.float32)
        except KeyError:
            extra_keypoints_2d = np.zeros((num_images, 19, 3), dtype=np.float32)
        keypoints_2d_all = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1)

        try:
            body_keypoints_3d = data["body_keypoints_3d"].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((num_images, 25, 4), dtype=np.float32)
        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0
        try:
            extra_keypoints_3d = data["extra_keypoints_3d"].astype(np.float32)
        except KeyError:
            import pdb
            print("Error while loading the 3D keypoints labels for {}".format(data["imgname"]))
            pdb.set_trace()
        keypoints_3d_all = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1)

        centers, bbox_sizes = [], []

        for n in range(num_images):
            imgname = data["imgname"][n]
            imgnames.append(imgname)

            image_file = os.path.join(self.img_dir, imgname)
            keypoints_2d_n = keypoints_2d_all[n]
            keypoints_3d_n = keypoints_3d_all[n]
            center_n = data["center"][n].copy()
            center_x = center_n[0]
            center_y = center_n[1]
            bbox_size_n = data["scale"][n]

            centers.append(center_n)
            bbox_sizes.append(bbox_size_n)

            if "body_pose" in data:
                body_pose_n = data["body_pose"][n]
            else:
                body_pose_n = np.zeros(72, dtype=np.float32)
            if "betas" in data:
                betas_n = data["betas"][n]
            else:
                betas_n = np.zeros(10, dtype=np.float32)
            if "has_body_pose" in data:
                has_body_pose_n = data["has_body_pose"][n]
            else:
                has_body_pose_n = 0.0
            if "has_betas" in data:
                has_betas_n = data["has_betas"][n]
            else:
                has_betas_n = 0.0

            smpl_params_n = {
                "global_orient": body_pose_n[:3],
                "body_pose": body_pose_n[3:],
                "betas": betas_n,
            }
            has_smpl_params_n = {
                "global_orient": has_body_pose_n,
                "body_pose": has_body_pose_n,
                "betas": has_betas_n,
            }
            smpl_params_is_axis_angle_n = {
                "global_orient": True,
                "body_pose": True,
                "betas": False,
            }

            img_patch_n, keypoints_2d_n, keypoints_3d_n, smpl_params_n, has_smpl_params_n, img_size_n = get_example(
                image_file,
                center_x,
                center_y,
                bbox_size_n,
                bbox_size_n,
                keypoints_2d_n,
                keypoints_3d_n,
                smpl_params_n,
                has_smpl_params_n,
                flip_kp_permutation=self.flip_keypoint_permutation,
                patch_width=self.img_size,
                patch_height=self.img_size,
                mean=self.mean,
                std=self.std,
                do_augment=False,
                augm_config=augm_config,
                load_image=True,
            )

            img_patch.append(img_patch_n)
            img_size.append(img_size_n)
            keypoints_2d.append(keypoints_2d_n)
            keypoints_3d.append(keypoints_3d_n)
            smpl_params.append(smpl_params_n)
            has_smpl_params.append(has_smpl_params_n)
            smpl_params_is_axis_angle.append(smpl_params_is_axis_angle_n)
        img_patch = np.stack(img_patch, axis=0)
        keypoints_2d = np.stack(keypoints_2d, axis=0)
        keypoints_3d = np.stack(keypoints_3d, axis=0)
        smpl_params = {
            k: np.stack([sp[k] for sp in smpl_params], axis=0)
            for k in smpl_params[0].keys()
        }
        has_smpl_params = {
            k: np.stack([sp[k] for sp in has_smpl_params], axis=0)
            for k in has_smpl_params[0].keys()
        }
        smpl_params_is_axis_angle = {
            k: np.stack([sp[k] for sp in smpl_params_is_axis_angle], axis=0)
            for k in smpl_params_is_axis_angle[0].keys()
        }
        img_size = np.stack(img_size, axis=0)

        item = {}
        item["img"] = torch.from_numpy(img_patch)
        item["keypoints_2d"] = torch.from_numpy(keypoints_2d.astype(np.float32))
        item["keypoints_3d"] = torch.from_numpy(keypoints_3d.astype(np.float32))
        item["smpl_params"] = {
            k: torch.from_numpy(v).float() for k, v in smpl_params.items()
        }
        item["has_smpl_params"] = {
            k: torch.from_numpy(v).bool() for k, v in has_smpl_params.items()
        }
        item["smpl_params_is_axis_angle"] = {
            k: torch.from_numpy(v).bool() for k, v in smpl_params_is_axis_angle.items()
        }
        item["imgname"] = imgnames
        item["box_center"] = torch.from_numpy(np.array(centers).astype(np.float32))
        item["box_size"] = torch.from_numpy(np.array(bbox_sizes).astype(np.float32))
        item["img_size"] = torch.from_numpy(1.0 * img_size[:, ::-1].astype(np.float32))

        # Regression estimates.
        if self.load_predicitons is not None:
            item["pred_betas"] = torch.from_numpy(self.predictions[idx]["pred_betas"])
            item["pred_pose"] = torch.from_numpy(self.predictions[idx]["pred_pose"])
            item["pred_cam"] = torch.from_numpy(self.predictions[idx]["pred_cam"])
            item["pred_cam_t"] = torch.from_numpy(self.predictions[idx]["pred_cam_t"])

        # OpenPose keypoints in full frame.
        if "body_keypoints_2d" in self.data[idx]:
            item["orig_keypoints_2d"] = torch.from_numpy(
                np.concatenate((self.data[idx]["body_keypoints_2d"], np.zeros((num_images, 19, 3))), axis=1).astype(np.float32)
            )

        return item