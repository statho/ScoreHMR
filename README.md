# ScoreHMR: Score-Guided Human Mesh Recovery

Code repository for the paper:
**Score-Guided Diffusion for 3D Human Recovery**\
[Anastasis Stathopoulos](https://statho.github.io/), [Ligong Han](https://phymhan.github.io/), [Dimitris Metaxas](https://people.cs.rutgers.edu/~dnm/)

[![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)](https://arxiv.org/abs/2403.09623)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://statho.github.io/ScoreHMR/)

![teaser](assets/teaser.jpg)


## Installation and Setup
First, clone the repository and submodules. Then, set up a new conda environment and install all dependencies, as follows:
```bash
git clone --recursive https://github.com/statho/ScoreHMR.git
cd ScoreHMR
source install_environment.sh
```

Download the pretrained model weights, and annotations for the datasets by running the following:
```
source download_data.sh
```
This will download all necessary data files, and place them in `data/`. Alternatively, you can download them from [here](https://drive.google.com/file/d/1W53UMg8kee3HGRTNd2aNhMUew_kj36OH/view?usp=sharing) and [here](https://drive.google.com/file/d/1f-D3xhQPMC9rwtaCVNoxtD4BQh4oQbY9/view?usp=sharing). Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de). Please go to the corresponding website and register to get access to the downloads section. Download the model, create a folder `data/smpl`, rename `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl`, and place it in `data/smpl/`.

Finally, if you wish to run the evaluation and/or training code, you will need to download the images/videos for the datasets. The instructions are mostly common with the description in [here](https://github.com/nkolot/ProHMR/blob/master/dataset_preprocessing/README.md). We provide the annotations for all datasets, so you will only need to download the images/videos. Edit the `IMG_DIR` in `score_hmr/configs/datasets.yml` accordingly.


## Run demo on images
The following command will run ScoreHMR on top of HMR 2.0, using detected keypoints from ViTPose and bounding boxes from ViTDet, on all images in the specified `--img_folder`. For each image, it will save a rendering of all the reconstructed people together in the front view.
```bash
python demo_image.py \
    --img_folder example_data/images \
    --out_folder demo_out/images
```


## Run demo on videos
The following command will first run tracking with 4D-Humans and 2D keypoint detection with ViTPose, and then run temporal model fitting with ScoreHMR on the video specified with `--input_video`. It will create a video rendering of the reconstructed people in the folder specified by `--output_folder`. It will also save intermediate results from 4D-Humans and ViTPose.
```bash
python demo_video.py \
    --input_video example_data/videos/breakdancing.mp4 \
    --out_folder demo_out/videos
```


## Evaluation
The evaluation code is contained in `eval/`. We provide evaluation on 3 different settings with the following scripts:
- `eval_keypoint_fitting.py` is used in single-frame model fitting evaluation as in Tables 1, 2 & 5 of the paper.
- `eval_multiview.py` is used to evaluate the multi-view refienement as in Table 3 of the paper.
- `eval_video.py` is used to evaluate ScoreHMR in temporal model fitting as in Table 4 of the paper.

The evaluation code uses cached HMR 2.0 predictions, which can be downloaded from [here](https://drive.google.com/file/d/1m9lv9uDYosIVZ-u0R3GCy1J1wHYNVUMP/view?usp=sharing) or by running:
```bash
source download_hmr2_preds.sh
```
We also provide example code for saving the HMR 2.0 predictions in the appropriate format in `data_preprocessing/cache_hmr2_preds.py`.

Evaluation code example:
```bash
python eval/eval_keypoint_fitting.py --dataset 3DPW-TEST --shuffle --use_default_ckpt
```
Running the above command will compute the MPJPE and Reconstruction Error before and after single-frame model fitting with ScoreHMR on the test set of 3DPW.


## Training
The training code uses cached image features. First, extract the PARE image features for the training datasets:
```
python data_preprocessing/cache_pare_preds.py
```
Then, start training using the following command:
```
python train.py --name <name_of_experiment>
```
Checkpoints and logs will be saved to `logs/`.


## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [ProHMR](https://github.com/nkolot/ProHMR)
- [PARE](https://github.com/mkocabas/PARE)
- [SLAHMR](https://github.com/vye16/slahmr)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{stathopoulos2024score,
  title  = {Score-Guided Diffusion for 3D Human Recovery},
  author = {Stathopoulos, Anastasis and Han, Ligong and Metaxas, Dimitris},
  booktitle = {CVPR},
  year = {2024}
}
```
