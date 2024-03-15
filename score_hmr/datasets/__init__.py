"""
Code borrowed from
https://github.com/nkolot/ProHMR/blob/master/prohmr/datasets/__init__.py
"""
from typing import Dict
import torch
import numpy as np
from yacs.config import CfgNode
from score_hmr.configs import to_lower
from .dataset import Dataset
from .image_dataset import ImageDataset
from .batched_image_dataset import BatchedImageDataset


def create_dataset(cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> Dataset:
    """
    Instantiate a dataset from a config file.
    Args:
        cfg (CfgNode): Model configuration file.
        dataset_cfg (CfgNode): Dataset configuration info.
        train (bool): Variable to select between train and val datasets.
    """
    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type(cfg, **to_lower(dataset_cfg), train=train)


class MixedDataset:
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> None:
        """
        Setup Mixed dataset containing different dataset mixed together.
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            train (bool): Variable to select between train and val datasets.
        """

        dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        self.datasets = []
        for dataset, v in dataset_list.items():
            print("=> Loading Dataset {} (Weight : {})".format(dataset, v.WEIGHT))
            self.datasets.append(create_dataset(cfg, dataset_cfg[dataset], train=train))
        self.weights = np.array(
            [v.WEIGHT for dataset, v in dataset_list.items()]
        ).cumsum()

    def __len__(self) -> int:
        """
        Returns:
            int: Sum of the lengths of each dataset
        """
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, i: int) -> Dict:
        """
        Index an element from the dataset.
        This is done by randomly choosing a dataset using the mixing percentages
        and then randomly choosing from the selected dataset.
        Returns:
            Dict: Dictionary containing data and labels for the selected example
        """
        p = torch.rand(1).item()
        for i in range(len(self.datasets)):
            if p <= self.weights[i]:
                p = torch.randint(0, len(self.datasets[i]), (1,)).item()
                return self.datasets[i][p]