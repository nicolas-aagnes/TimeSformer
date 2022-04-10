# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from fileinput import hook_compressed
from stringprep import map_table_b3
import torch
import torch.nn as nn
import numpy as np
import copy

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}

_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

_HOMOGRAPHY_MATRICES = {
    "california": {
        "left_to_center": np.array(
            [
                [1.18659063e01, -1.03581832e00, -1.51728898e04],
                [2.65481365e00, 1.12722687e01, -7.32983168e03],
                [3.31518648e-03, -4.49825472e-05, 1.00000000e00],
            ]
        ),
        "center_to_left": np.array(
            [
                [1.50603313e-02, 2.36495722e-03, 2.45843485e02],
                [-3.70978096e-02, 8.55607906e-02, 6.42652164e01],
                [-5.15965607e-05, -3.99153192e-06, 1.87873813e-01],
            ]
        ),
        "center_to_right": np.array(
            [
                [1.41878778e00, 7.22723014e-03, -1.56081044e03],
                [3.71992232e-01, 1.16845780e00, -1.27144481e02],
                [6.61402508e-04, 1.14502912e-04, 1.00000000e00],
            ]
        ),
        "right_to_center": np.array(
            [
                [4.20269624e-01, -6.60572983e-02, 6.47562397e02],
                [-1.62025721e-01, 8.70763894e-01, -1.42178613e02],
                [-2.59414966e-04, -5.60145387e-05, 5.87980472e-01],
            ]
        ),
    }
}


class HomographyLoss:
    def __init__(self, homography_matrices_location, num_cameras):
        self.num_cameras = num_cameras

        self.m = copy.deepcopy(_HOMOGRAPHY_MATRICES[homography_matrices_location])

        for transf in self.m.keys():
            self.m[transf] = (
                torch.from_numpy(self.m[transf].copy()).to(_DEVICE).unsqueeze(0).float()
            )

    def __call__(self, embeddings, labels):
        assert 1 == 0, "Working on getting labels to work in the loss function."
        
        (batch_size, num_cameras, embedding_dim, space_dim) = embeddings.shape
        assert num_cameras == self.num_cameras, num_cameras
        assert space_dim == 3, space_dim
        
        if self.num_cameras == 3:
            z_c = embeddings[:, 0]
            z_l = embeddings[:, 1]
            z_r = embeddings[:, 2]

            loss = (
                torch.norm(z_c - torch.bmm(z_l, self.m["left_to_center"]), dim=(1, 2))
                + torch.norm(z_l - torch.bmm(z_c, self.m["center_to_left"]), dim=(1, 2))
                + torch.norm(z_c - torch.bmm(z_r, self.m["center_to_right"]), dim=(1, 2))
                + torch.norm(z_r - torch.bmm(z_c, self.m["right_to_center"]), dim=(1, 2))
            )
        
        elif self.num_cameras == 2:
            z_c = embeddings[:, 0]
            z_l = embeddings[:, 1]

            loss = torch.norm(z_c - torch.bmm(z_l, self.m["left_to_center"]), dim=(1, 2))
        else:
            raise NotImplementedError

        return loss.mean()


def get_loss_func(loss_name, homography_matrices_location, num_cameras):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
        cfg: experiment config.
    """

    if homography_matrices_location is not None:
        _LOSSES["homography_loss"] = HomographyLoss(homography_matrices_location, num_cameras)

    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
