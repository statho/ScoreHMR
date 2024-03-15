import numpy as np

OP_NUM_JOINTS = 25
body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

# Image constants.
HMR2_IMG_SIZE = 256
DEFAULT_IMG_SIZE = 224
DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])

RESULTS_DIR = 'logs'                    # directory to save checkpoints and logs during training.
CHECKPOINT_DIR = 'data/model_weights'   # directory with pretrained models.
PARE_CHECKPOINT = f"{CHECKPOINT_DIR}/pare/pare_checkpoint.ckpt"


# Keypoint defitition (44).

JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    "OP Nose",
    "OP Neck",
    "OP RShoulder",
    "OP RElbow",
    "OP RWrist",
    "OP LShoulder",
    "OP LElbow",
    "OP LWrist",
    "OP MidHip",
    "OP RHip",
    "OP RKnee",
    "OP RAnkle",
    "OP LHip",
    "OP LKnee",
    "OP LAnkle",
    "OP REye",
    "OP LEye",
    "OP REar",
    "OP LEar",
    "OP LBigToe",
    "OP LSmallToe",
    "OP LHeel",
    "OP RBigToe",
    "OP RSmallToe",
    "OP RHeel",
    # 19 Ground Truth joints (superset of joints from different datasets)
    "R Ankle",
    "R Knee",
    "R Hip",
    "L Hip",
    "L Knee",
    "L Ankle",
    "R Wrist",
    "R Elbow",
    "R Shoulder",
    "L Shoulder",
    "L Elbow",
    "L Wrist",
    "Neck (LSP)",
    "Top of Head (LSP)",
    "Pelvis (MPII)",
    "Thorax (MPII)",
    "Spine (H36M)",
    "Jaw (H36M)",
    "Head (H36M)",
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}


# SMPL joints (24).

SMPL_JOINTS = [
    "pelvis",
    "L_hip",
    "R_hip",
    "spine1",
    "L_knee",
    "R_knee",
    "spine2",
    "L_ankle",
    "R_ankle",
    "spine3",
    "L_foot",
    "R_foot",
    "neck",
    "L_collar",
    "R_collar",
    "head",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hand",
    "R_hand",
]