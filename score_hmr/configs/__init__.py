"""
Code borrowed and adapted from: https://github.com/nkolot/ProHMR/blob/master/prohmr/configs/__init__.py
"""
import os
from typing import Dict
from yacs.config import CfgNode as CN


def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}


_C = CN(new_allowed=True)

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.SHUFFLE = True
_C.TRAIN.WARMUP = False

_C.DATASETS = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
_C.MODEL.IMAGE_SIZE = 224

_C.EXTRA = CN(new_allowed=True)
_C.EXTRA.FOCAL_LENGTH = 5000

_C.DATASETS.CONFIG = CN(new_allowed=True)
_C.DATASETS.CONFIG.SCALE_FACTOR = 0.3
_C.DATASETS.CONFIG.ROT_FACTOR = 30
_C.DATASETS.CONFIG.TRANS_FACTOR = 0.02
_C.DATASETS.CONFIG.COLOR_SCALE = 0.2
_C.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
_C.DATASETS.CONFIG.TRANS_AUG_RATE = 0.5
_C.DATASETS.CONFIG.DO_FLIP = True
_C.DATASETS.CONFIG.FLIP_AUG_RATE = 0.5
_C.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.10


def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def dataset_config() -> CN:
    """
    Get dataset config file
    Returns:
      CfgNode: Dataset config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "datasets.yaml"
    )
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def model_config() -> CN:
    """
    Get model config file
    Returns:
      CfgNode: Model config as a yacs CfgNode object.
    """
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "base.yaml")
    cfg = default_config()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
