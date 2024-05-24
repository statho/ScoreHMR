import os
import cv2
import torch
import argparse
import shutil
import numpy as np
from pathlib import Path
import warnings
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full

from score_hmr.utils import *
from score_hmr.configs import model_config
from demo.videt_dataset import ViTDetDataset
from demo.vitpose_model import ViTPoseModel
from score_hmr.models.model_utils import load_diffusion_model, load_pare

warnings.filterwarnings('ignore')


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out/images', help='Output folder to save rendered results')
    args = parser.parse_args()

    # Download and load checkpoints.
    download_models(CACHE_DIR_4DHUMANS)
    # Copy SMPL model to the appropriate path for HMR 2.0 if it does not exist.
    if not os.path.isfile(f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl'):
        shutil.copy('data/smpl/SMPL_NEUTRAL.pkl', f'{CACHE_DIR_4DHUMANS}/data/smpl/')
    hmr2_model, model_cfg_hmr2 = load_hmr2(DEFAULT_CHECKPOINT)

    # Setup HMR2.0 model.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    hmr2_model = hmr2_model.to(device)
    hmr2_model.eval()

    # Set up keypoint detector.
    kp_detector = ViTPoseModel(device)

    # Load human detector.
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg_hmr2, faces=hmr2_model.smpl.faces)
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=5000.,
    )

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
        "use_default_ckpt": True,
        "device": device,
    }
    diffusion_model = load_diffusion_model(model_cfg, **extra_args)


    # Make output directory if it does not exist.
    os.makedirs(args.out_folder, exist_ok=True)


    # Iterate over all images in folder.
    for img_path in Path(args.img_folder).glob('*.jpg'):
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image.
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect keypoints for each person.
        vitposes_out = kp_detector.predict_pose(
            img_cv2[:, :, ::-1],
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        vitposes_list = []
        for vitpose in vitposes_out:
            vitpose_2d = np.zeros([25, 3])
            vitpose_2d[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = (
                vitpose["keypoints"]
            )
            vitposes_list.append(vitpose_2d)
        body_keypoints_2d = np.stack(vitposes_list)


        # Create separate dataset of HMR 2.0 and ScoreHMR, since the input should be of different resolution.
        dataset_hmr2 = ViTDetDataset(model_cfg_hmr2, img_cv2, pred_bboxes)
        dataloader_hmr2 = torch.utils.data.DataLoader(dataset_hmr2, batch_size=pred_bboxes.shape[0], shuffle=False, num_workers=0)

        dataset = ViTDetDataset(model_cfg_hmr2, img_cv2, pred_bboxes, body_keypoints_2d, is_hmr2=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=pred_bboxes.shape[0], shuffle=False, num_workers=0)

        # Get predictions from HMR 2.0
        hmr2_batch = recursive_to(next(iter(dataloader_hmr2)), device)
        with torch.no_grad():
            out = hmr2_model(hmr2_batch)

        pred_cam = out['pred_cam']
        batch_size = pred_cam.shape[0]
        box_center = hmr2_batch["box_center"].float()
        box_size = hmr2_batch["box_size"].float()
        img_size = hmr2_batch["img_size"].float()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size)

        hmr2_verts = out['pred_vertices'].cpu().numpy()
        hmr2_cam_t = pred_cam_t_full.cpu().numpy()


        # Run iterative refinement with ScoreHMR.
        batch = recursive_to(next(iter(dataloader)), device)
        with torch.no_grad():
            pare_out = pare(batch["img"], get_feats=True)
        cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
        cond_feats = img_feat_standarizer(cond_feats) # normalize image features

        # Prepare things for model fitting.
        batch["camera_center"] = batch["img_size"] / 2
        batch["joints_2d"] = batch["keypoints_2d"][:, :, :2]
        batch["joints_conf"] = batch["keypoints_2d"][:, :, [2]]
        batch["focal_length"] = model_cfg.EXTRA.FOCAL_LENGTH * torch.ones(
                batch_size,
                2,
                device=device,
                dtype=batch["keypoints_2d"].dtype,
            )
        batch['pred_betas'] = out['pred_smpl_params']['betas']
        batch['pred_pose'] = torch.cat((out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose']), axis=1)
        batch["init_cam_t"] = pred_cam_t_full

        # Run ScoreHMR.
        print(f'=> Running ScoreHMR for image: {img_fn}')
        with torch.no_grad():
            dm_out = diffusion_model.sample(
                batch, cond_feats, batch_size=batch_size
            )

        pred_smpl_params = prepare_smpl_params(
            dm_out['x_0'],
            num_samples = 1,
            use_betas = False,
            pred_betas=batch["pred_betas"],
        )
        smpl_out = diffusion_model.smpl(**pred_smpl_params, pose2rot=False)

        opt_verts = smpl_out.vertices.cpu().numpy()
        opt_cam_t = dm_out['camera_translation'].cpu().numpy()


        # Render front view.
        print(f'=> Rendering image: {img_fn}')
        render_res = img_size[0].cpu().numpy()
        cam_view = renderer.render_rgba_multiple(opt_verts, cam_t=opt_cam_t, render_res=render_res, **misc_args)

        # Overlay and save image.
        input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
        cv2.imwrite(os.path.join(args.out_folder, f'opt_{img_fn}.png'), 255*input_img_overlay[:, :, ::-1])



if __name__ == '__main__':
    main()