import torch
import numpy as np


def cycle(dl):
    """
    Generate data from dataloader.
    """
    while True:
        for data in dl:
            yield data


def filter_based_on_pose(batch):
    """
    Keep only samples with pseudo-GT pose annotations.
    """
    has_smpl_params = batch["has_smpl_params"]["body_pose"] > 0
    batch_size = has_smpl_params.sum().item()
    for key in batch:
        if key == "imgname":
            batch["imgname"] = list(
                np.array(batch["imgname"])[has_smpl_params.cpu().numpy()]
            )
        elif key in ["smpl_params", "has_smpl_params", "smpl_params_is_axis_angle"]:
            for nested_key in batch[key]:
                batch[key][nested_key] = batch[key][nested_key][has_smpl_params]
        else:
            batch[key] = batch[key][has_smpl_params]
    assert (
        batch["img"].size(0) == batch["smpl_params"]["body_pose"].size(0) == batch_size
        and torch.all(batch["has_smpl_params"]["body_pose"]).item()
        and torch.all(batch["has_smpl_params"]["global_orient"]).item()
    ), "Error in discarding images with no SMPL pseudo-GT"
    return batch
