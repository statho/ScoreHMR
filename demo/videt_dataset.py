"""
Code adapted from
https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/datasets/vitdet_dataset.py
"""
import torch
import cv2
import numpy as np
from typing import Dict
from yacs.config import CfgNode
from constants import DEFAULT_MEAN, DEFAULT_STD, HMR2_IMG_SIZE, DEFAULT_IMG_SIZE
from hmr2.datasets.utils import convert_cvimg_to_tensor, expand_to_aspect_ratio, generate_image_patch_cv2


class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            cfg: CfgNode,
            img_cv2: np.array,
            boxes: np.array,
            body_keypoints: np.array = None,
            train: bool = False,
            is_hmr2: bool = True,
            **kwargs
        ):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.is_hmr2 = is_hmr2
        self.img_size = HMR2_IMG_SIZE if is_hmr2 else DEFAULT_IMG_SIZE

        # Preprocess annotations
        N = boxes.shape[0]
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.keypoints_2d = np.concatenate( (body_keypoints, np.zeros((N, 19, 3))), axis=1).astype(np.float32) if not is_hmr2 else None

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        if self.is_hmr2:
            BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
            bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        else:
            bbox_size = (200 * scale).max()

        patch_width = patch_height = self.img_size

        # generate image patch
        cvimg = self.img_cv2.copy()
        img_patch_cv, _ = generate_image_patch_cv2(
            cvimg,
            center_x, center_y,
            bbox_size, bbox_size,
            patch_width, patch_height,
            False, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT
        )
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        if not self.is_hmr2:
            item['keypoints_2d'] = self.keypoints_2d[idx].copy()
        return item
