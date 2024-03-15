from .hrnet_pare import *


def get_backbone_info(backbone):
    info = {
        "hrnet_w32": {"n_output_channels": 480, "downsample_rate": 4},
        "hrnet_w48": {"n_output_channels": 720, "downsample_rate": 4},
    }
    return info[backbone]
