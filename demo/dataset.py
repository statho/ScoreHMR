"""
Code borrowed and adapted from:
https://github.com/vye16/slahmr/blob/main/slahmr/data/dataset.py
"""
import os
import typing
import imageio
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from .data_utils import read_keypoints, load_smpl_cam_box, read_images


SHOT_PAD = 0
MIN_SEQ_LEN = 40
MAX_NUM_TRACKS = 5
MIN_TRACK_LEN = 40
MIN_KEYP_CONF = 0.4


class MultiPeopleDataset(Dataset):
    def __init__(
        self,
        data_sources: typing.Dict,
        seq_name,
        tid_spec="all",
        shot_idx=0,
        start_idx=0,
        end_idx=-1,
        pad_shot=False,
    ):
        print('=> Preparing dataset ...')
        self.seq_name = seq_name
        self.data_sources = data_sources
        self.shot_idx = shot_idx

        # select only images in the desired shot
        img_files, _ = get_shot_img_files(
            self.data_sources["shots"], shot_idx, pad_shot
        )
        end_idx = end_idx if end_idx > 0 else len(img_files)
        self.data_start, self.data_end = start_idx, end_idx
        img_files = img_files[start_idx:end_idx]
        self.img_names = [get_name(f) for f in img_files]
        self.num_imgs = len(self.img_names)

        img_dir = self.data_sources["images"]
        assert os.path.isdir(img_dir)
        self.img_paths = [os.path.join(img_dir, f) for f in img_files]

        img_h, img_w = imageio.imread(self.img_paths[0]).shape[:2]
        self.img_size = img_w, img_h
        print(f"USING TOTAL {self.num_imgs} {img_w}x{img_h} IMGS")

        # find the tracks in the video
        track_root = self.data_sources["tracks"]
        if tid_spec == "all" or tid_spec.startswith("longest"):
            n_tracks = MAX_NUM_TRACKS
            if tid_spec.startswith("longest"):
                n_tracks = int(tid_spec.split("-")[1])
            # get the longest tracks in the selected shot
            track_ids = sorted(os.listdir(track_root))
            track_paths = [
                [f"{track_root}/{tid}/{name}_keypoints.json" for name in self.img_names]
                for tid in track_ids
            ]
            track_lens = [
                len(list(filter(os.path.isfile, paths))) for paths in track_paths
            ]
            track_ids = [
                track_ids[i]
                for i in np.argsort(track_lens)[::-1]
                if track_lens[i] > MIN_TRACK_LEN
            ]
            print("TRACK LENGTHS", track_lens)
            track_ids = track_ids[:n_tracks]
        else:
            track_ids = [f"{int(tid):03d}" for tid in tid_spec.split("-")]

        print("TRACK IDS", track_ids)

        self.track_ids = track_ids
        self.n_tracks = len(track_ids)
        self.track_dirs = [os.path.join(track_root, tid) for tid in track_ids]

        # keep a list of frame index masks of whether a track is available in a frame
        sidx = np.inf
        eidx = -1
        self.track_vis_masks = []
        for pred_dir in self.track_dirs:
            kp_paths = [f"{pred_dir}/{x}_keypoints.json" for x in self.img_names]
            has_kp = [os.path.isfile(x) for x in kp_paths]

            # keep track of which frames this track is visible in
            vis_mask = np.array(has_kp)
            idcs = np.where(vis_mask)[0]
            if len(idcs) > 0:
                si, ei = min(idcs), max(idcs)
                sidx = min(sidx, si)
                eidx = max(eidx, ei)
            self.track_vis_masks.append(vis_mask)

        eidx = max(eidx + 1, 0)
        sidx = min(sidx, eidx)
        # print("GLOBAL START", sidx)
        # print("GLOBAL END", eidx)
        self.start_idx = sidx
        self.end_idx = eidx
        self.seq_len = eidx - sidx
        self.seq_intervals = [(sidx, eidx) for _ in track_ids]

        self.sel_img_paths = self.img_paths[sidx:eidx]
        self.sel_img_names = self.img_names[sidx:eidx]

        # used to cache data
        self.data_dict = {}

    def __len__(self):
        return self.n_tracks

    def load_data(self, interp_input=True):
        if len(self.data_dict) > 0:
            return

        # get data for each track
        data_out = {
            "joints2d": [],
            "vis_mask": [],
            "track_interval": [],
            "init_body_pose": [],
            "init_root_orient": [],
            "init_trans": [],
            "img": [],
            "box": [],
            "box_center": [],
            "box_scale": [],
            "img_size": [],
            "pred_betas": [],
            "cam_bbox": [],
        }

        # create batches of sequences
        # each batch is a track for a person
        T = self.seq_len
        sidx, eidx = self.start_idx, self.end_idx
        for i, tid in enumerate(self.track_ids):
            # load mask of visible frames for this track
            vis_mask = self.track_vis_masks[i][sidx:eidx]  # (T)
            vis_idcs = np.where(vis_mask)[0]
            track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
            data_out["track_interval"].append([track_s, track_e])

            vis_mask = get_ternary_mask(vis_mask)
            data_out["vis_mask"].append(vis_mask)

            # load 2d keypoints for visible frames
            kp_paths = [
                f"{self.track_dirs[i]}/{x}_keypoints.json" for x in self.sel_img_names
            ]
            # (T, J, 3) (x, y, conf)
            joints2d_data = np.stack(
                [read_keypoints(p) for p in kp_paths], axis=0
            ).astype(np.float32)
            # Discard bad ViTPose detections
            joints2d_data[
                np.repeat(joints2d_data[:, :, [2]] < MIN_KEYP_CONF, 3, axis=2)
            ] = 0
            data_out["joints2d"].append(joints2d_data)

            # load single image smpl predictions
            pred_paths = [
                f"{self.track_dirs[i]}/{x}_smpl.json" for x in self.sel_img_names
            ]

            # Load SMPL-preds, camera, and bounding box information for each tracklet.
            # Intrpolate prediction for occluded trakclets (vis=0) from the detected tracklets (vis=1).
            init_predictions = load_smpl_cam_box(pred_paths, interp=interp_input)

            images = np.stack([
                    read_images(img_path, box_c, box_s)
                    for (img_path, box_c, box_s) in zip(self.sel_img_paths, init_predictions["box_center"], init_predictions["box_scale"])
            ])

            data_out["img"].append(images)
            data_out["box"].append(init_predictions["box"])
            data_out["box_center"].append(init_predictions["box_center"])
            data_out["box_scale"].append(init_predictions["box_scale"])
            data_out["cam_bbox"].append(init_predictions["cam_bbox"])
            data_out["init_trans"].append(init_predictions["cam_trans"])
            data_out["pred_betas"].append(init_predictions["betas"])
            data_out["init_body_pose"].append(init_predictions["body_pose"])
            data_out["init_root_orient"].append(init_predictions["global_orient"])

        self.data_dict = data_out

    def __getitem__(self, idx):
        if len(self.data_dict) < 1:
            self.load_data()

        obs_data = dict()
        # the frames the track is visible in
        obs_data["vis_mask"] = torch.Tensor(self.data_dict["vis_mask"][idx])
        # the frames used in this subsequence
        obs_data["seq_interval"] = torch.Tensor(list(self.seq_intervals[idx])).to(
            torch.int
        )
        # the start and end interval of available keypoints
        obs_data["track_interval"] = torch.Tensor(
            self.data_dict["track_interval"][idx]
        ).int()
        obs_data["seq_name"] = self.seq_name
        obs_data["track_id"] = int(self.track_ids[idx])
        obs_data["img"] = torch.Tensor(self.data_dict["img"][idx])
        obs_data["box"] = torch.Tensor(self.data_dict["box"][idx])
        obs_data["box_center"] = torch.Tensor(self.data_dict["box_center"][idx])
        obs_data["box_scale"] = torch.Tensor(self.data_dict["box_scale"][idx])
        obs_data["cam_bbox"] = torch.Tensor(self.data_dict["cam_bbox"][idx])
        obs_data["pred_betas"] = torch.Tensor(self.data_dict["pred_betas"][idx])
        obs_data["init_body_pose"] = torch.Tensor(self.data_dict["init_body_pose"][idx])
        obs_data["init_root_orient"] = torch.Tensor(self.data_dict["init_root_orient"][idx])
        obs_data["pred_cam_t"] = torch.Tensor(self.data_dict["init_trans"][idx])
        obs_data["keypoints_2d"] = torch.Tensor(self.data_dict["joints2d"][idx])
        return obs_data



def get_name(x):
    return os.path.splitext(os.path.basename(x))[0]


def get_shot_img_files(shots_path, shot_idx, shot_pad=SHOT_PAD):
    assert os.path.isfile(shots_path)
    with open(shots_path, "r") as f:
        shots_dict = json.load(f)
    img_names = sorted(shots_dict.keys())
    N = len(img_names)
    shot_mask = np.array([shots_dict[x] == shot_idx for x in img_names])

    idcs = np.where(shot_mask)[0]
    if shot_pad > 0:  # drop the frames before/after shot change
        if min(idcs) > 0:
            idcs = idcs[shot_pad:]
        if len(idcs) > 0 and max(idcs) < N - 1:
            idcs = idcs[:-shot_pad]
        if len(idcs) < MIN_SEQ_LEN:
            raise ValueError("shot is too short for optimization")

        shot_mask = np.zeros(N, dtype=bool)
        shot_mask[idcs] = 1
    sel_paths = [img_names[i] for i in idcs]
    print(f"FOUND {len(idcs)}/{len(shots_dict)} FRAMES FOR SHOT {shot_idx}")
    return sel_paths, idcs


def get_ternary_mask(vis_mask):
    # get the track start and end idcs relative to the filtered interval
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
    # -1 = track out of scene, 0 = occlusion, 1 = visible
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1
    vis_mask[track_e:] = -1
    return vis_mask
