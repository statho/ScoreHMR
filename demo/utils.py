"""
Code adapted from:
https://github.com/vye16/slahmr/blob/main/slahmr/preproc/launch_phalp.py
"""

import os
import cv2
import torch
import subprocess
import numpy as np
from phalp.utils.utils import progress_bar
from .export_phalp import export_sequence_results


def slice_dict(d, idx):
    out = d.copy()
    for k, v in d.items():
        # if not isinstance(v, torch.Tensor) or v.ndim < 3:
        #     continue
        out[k] = v[idx]
    return out


def slice_dict_start_end(d, start, end):
    out = d.copy()
    for k, v in d.items():
        if not isinstance(v, torch.Tensor) or v.dim() < 2:
            continue
        out[k] = v[start:end]
    return out


def video_to_frames(
    path,
    out_dir,
    fps=30,
    ext="jpg",
    down_scale=1,
    start_sec=0,
    end_sec=-1,
    overwrite=False,
    **kwargs,
):
    """
    Extract image frames from the given video.
    """
    os.makedirs(out_dir, exist_ok=True)

    arg_str = f"-copyts -qscale:v 2 -vf fps={fps}"
    if down_scale != 1:
        arg_str = f"{arg_str},scale='iw/{down_scale}:ih/{down_scale}'"
    if start_sec > 0:
        arg_str = f"{arg_str} -ss {start_sec}"
    if end_sec > start_sec:
        arg_str = f"{arg_str} -to {end_sec}"

    yn = "-y" if overwrite else "-n"
    cmd = f"ffmpeg -i {path} {arg_str} {out_dir}/%06d.{ext} {yn}"
    print(cmd)

    return subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)


def create_visuals(
    renderer,
    verts,
    cam_trans,
    img_paths,
):
    frame_list = []
    for t_, img_path in progress_bar(enumerate(img_paths), description=f"Rendering ...", total=len(img_paths), disable=False):
        image = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        height, width, _ = image.shape
        render_res = [width, height]
        # Keep input image to overlay in top-left corner.
        scaled_height, scaled_width = height // 4, width // 4
        input_image = cv2.resize(image.copy(), (scaled_width, scaled_height))

        # Render all people in the scene.
        cam_view = renderer.render_rgba_multiple(
            vertices=verts[:, t_],
            cam_t=cam_trans[:, t_],
            render_res=render_res,
        )
        image = np.concatenate([image, np.ones_like(image[:, :, :1])], axis=2)  # add alpha channel
        # Combine input image and rendered meshes.
        image = image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        # Add black layout.
        image[: scaled_height + 5, : scaled_width + 5, :3] = 0.0
        # Add input image to the top-left corner.
        image[:scaled_height, :scaled_width, :3] = input_image

        image = 255 * image[:, :, ::-1]
        frame_list.append(image)

    assert len(frame_list) > 0, "No valid frame to save."
    return frame_list



def launch_phalp(img_dir, res_dir, overwrite=False):
    """
    Run tracking.
    """
    cmd_args = [
        "python demo/track.py",
        f"video.source={img_dir}",
        f"video.output_dir={res_dir}",
        f"overwrite={overwrite}",
        "detect_shots=True",
        "video.extract_video=False",
        "render.enable=False",
    ]
    cmd = " ".join(cmd_args)
    print(cmd)
    return subprocess.call(cmd, shell=True)


def process_seq(
    out_root,
    seq,
    img_dir,
    out_name="phalp_out",
    track_name="track_preds",
    shot_name="shot_idcs",
    overwrite=False,
):
    """
    Run and export tracking results.
    """
    name = os.path.basename(seq).split(".")[0]
    res_root = f"{out_root}/{out_name}/{name}"
    os.makedirs(res_root, exist_ok=True)
    res_dir = os.path.join(res_root, "results")
    res_path = f"{res_root}/{name}.pkl"

    # Paths to export the PHALP predictions.
    track_dir = f"{out_root}/{track_name}/{name}"
    shot_path = f"{out_root}/{shot_name}/{name}.json"

    # Run PHALP tracking.
    if overwrite or not os.path.isfile(res_path):
        res = launch_phalp(img_dir, res_root, overwrite)
        assert res == 0, "PHALP FAILED"
        os.rename(f"{res_dir}/demo_{name}.pkl", res_path)
        # Remove empty directories.
        empty_dirs = ["_DEMO", "results", "results_tracks", "_TMP"]
        for dir_ in empty_dirs:
            os.rmdir(f"{res_root}/{dir_}")

        # Create necessary files for the demo.
        export_sequence_results(res_path, track_dir, shot_path)
