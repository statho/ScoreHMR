import os
import cv2
import json
import argparse
import warnings
from torch.utils.data import DataLoader

from demo.utils import *
from demo.dataset import MultiPeopleDataset
from score_hmr.utils import *
from score_hmr.configs import model_config
from score_hmr.utils.geometry import aa_to_rotmat
from score_hmr.models.model_utils import load_diffusion_model, load_pare
from score_hmr.utils.mesh_renderer import MeshRenderer

warnings.filterwarnings('ignore')


NUM_SAMPLES = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="example_data/breakdancing.mp4", help="Path of the input video.")
    parser.add_argument("--out_folder", type=str, default="demo_out/videos", help="Path to save the output video.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Indicate whether or not to overwrite the 4D-Human tracklets.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate to save the output video.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    OUT_DIR = args.out_folder

    video_name = os.path.basename(args.input_video)
    filename, _ = os.path.splitext(video_name)
    img_dir = f'{OUT_DIR}/images/{filename}'

    # Extract the frames of the input video.
    if not os.path.isdir(img_dir):
        video_to_frames(path=args.input_video, out_dir=img_dir)

    # Detects shots, runs 4D-Humans tracking, detects 2D with ViTPose, and prepares all necessary files.
    process_seq(out_root=OUT_DIR, seq=args.input_video, img_dir=img_dir, overwrite=args.overwrite)

    # Get the number of shots in the video.
    shots_path = f'{OUT_DIR}/shot_idcs/{filename}.json'
    with open(shots_path, "r") as f:
        shots_dict = json.load(f)
    num_shots = max(shots_dict.values())


    # ----------------------------------------
    ### Prepare ScoreHMR ###

    # Load config.
    model_cfg = model_config()

    # Load PARE model.
    pare = load_pare(model_cfg.SMPL).to(device)
    pare.eval()

    img_feat_standarizer = StandarizeImageFeatures(
        backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
        use_betas=False,
        device=device,
    )

    # Load diffusion model.
    extra_args = {
        "keypoint_guidance": True,
        "temporal_guidance": True,
        "use_default_ckpt": True,
        "device": device,
    }
    diffusion_model = load_diffusion_model(model_cfg, **extra_args)

    # ----------------------------------------


    # Set up renderer.
    renderer = MeshRenderer(model_cfg, faces=diffusion_model.smpl.faces)


    ## Iterate over shots in the video ##

    for shot_idx in range(num_shots+1):
        data_sources = {
            "images": f"{OUT_DIR}/images/{filename}",
            "tracks": f"{OUT_DIR}/track_preds/{filename}",
            "shots": f"{OUT_DIR}/shot_idcs/{filename}.json",
        }

        # Create dataset.
        dataset = MultiPeopleDataset(data_sources=data_sources, seq_name=filename, shot_idx=shot_idx)

        # Ignore shots with no tracklets or not long enough tracklets.
        if len(dataset.track_ids) <= 0 or dataset.num_imgs <= 20:
            continue

        B = len(dataset)     # number of people
        T = dataset.seq_len  # number of frames
        loader = DataLoader(dataset, batch_size=B, shuffle=False)

        obs_data = recursive_to(next(iter(loader)), device)

        num_tracks = obs_data["track_id"].size(0)
        pred_cam_t_all = torch.zeros((B, T, 3))
        pred_vertices_all = torch.zeros((B, T, 6890, 3))


        ## Iterate over tracklets ##

        for track_idx in range(num_tracks):
            # Get the the data for the current tracklet.
            batch = slice_dict(obs_data, track_idx)
            start_idx, end_idx = batch["track_interval"]
            start_idx = start_idx.item()
            end_idx = end_idx.item()

            # Keep only the valid data of the tracklet.
            batch = slice_dict_start_end(batch, start_idx, end_idx)
            batch_size = batch["keypoints_2d"].size(0)
            batch["img_size"] = (
                torch.Tensor(dataset.img_size)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .to(device)
            )
            batch["camera_center"] = batch["img_size"] / 2
            global_orient_rotmat = aa_to_rotmat(batch["init_root_orient"]).reshape(batch_size, -1, 3, 3)
            body_pose_rotmat = aa_to_rotmat(batch["init_body_pose"].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
            batch["pred_pose"] = torch.cat((global_orient_rotmat, body_pose_rotmat), dim=1)
            focal_length = model_cfg.EXTRA.FOCAL_LENGTH * torch.ones(
                batch_size,
                2,
                device=device,
                dtype=batch["keypoints_2d"].dtype,
            )
            batch["focal_length"] = focal_length

            # Get PARE image features.
            with torch.no_grad():
                pare_out = pare(batch["img"], get_feats=True)
            cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
            cond_feats = img_feat_standarizer(cond_feats) # normalize image features

            batch["init_cam_t"] = batch["pred_cam_t"]
            batch["joints_2d"] = batch["keypoints_2d"][:, :, :2]
            batch["joints_conf"] = batch["keypoints_2d"][:, :, [2]]

            # Iterative refinement with ScoreHMR.
            print(f'=> Running ScoreHMR for tracklet {track_idx+1}/{num_tracks}')
            with torch.no_grad():
                dm_out = diffusion_model.sample(
                    batch, cond_feats, batch_size=batch_size * NUM_SAMPLES
                )

            pred_smpl_params = prepare_smpl_params(
                dm_out['x_0'],
                num_samples = NUM_SAMPLES,
                use_betas = False,
                pred_betas=batch["pred_betas"],
            )
            smpl_out = diffusion_model.smpl(**pred_smpl_params, pose2rot=False)
            pred_cam_t_all[track_idx, start_idx:end_idx] = dm_out['camera_translation'].cpu()
            pred_vertices_all[track_idx, start_idx:end_idx] = smpl_out.vertices.cpu()


        # Save output video.
        frame_list = create_visuals(
            renderer,
            pred_vertices_all.numpy(),
            pred_cam_t_all.numpy(),
            dataset.sel_img_paths,
        )

        height, width, _ = frame_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(f"{OUT_DIR}", exist_ok=True)
        video_writer = cv2.VideoWriter(
            f"{OUT_DIR}/{filename}_{shot_idx}.mp4",
            fourcc,
            args.fps,
            (width, height),
        )
        for frame in frame_list:
            video_writer.write(cv2.convertScaleAbs(frame))
        video_writer.release()


if __name__ == "__main__":
    main()