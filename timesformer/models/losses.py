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
    "no_truck": {
        "left_to_center": np.array(
            [
                [-7.45570216e00, 2.50052903e-01, 1.22998523e04],
                [-1.59886298e00, -7.65049176e00, 4.97236885e03],
                [-3.07148730e-03, 2.99949796e-04, 1.00000000e00],
            ]
        ),
        "center_to_left": np.array(
            [
                [3.97156887e-02, -1.49414012e-02, -4.14202948e02],
                [5.94031292e-02, -1.31733814e-01, -7.56206025e01],
                [1.04168277e-04, -6.37879319e-06, -2.49536708e-01],
            ]
        ),
        "center_to_right": np.array(
            [
                [1.84669078e00, -3.04335377e-02, -3.02779889e03],
                [6.91965454e-01, 1.57569186e00, -4.03583077e02],
                [1.01035547e-03, -2.77860842e-05, 1.00000000e00],
            ]
        ),
        "right_to_center": np.array(
            [
                [2.00546478e-01, 1.46856979e-02, 6.13141303e02],
                [-1.40971341e-01, 6.28867785e-01, -1.73032474e02],
                [-2.06540274e-04, 2.63599790e-06, 3.75701433e-01],
            ]
        ),
    },
    "white_truck": {
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
    },
}

# resize = (224, 224)
# shape = cv2.imread(left_img_path).shape
# _SCALE = np.array([[resize[0] / shape[1], 0, 0], [0, resize[1] / shape[0], 0], [0, 0, 1]])
_SCALE = np.array(  # (1216 x 1936) --> (224 x 224)
    [
        [0.11570248, 0, 0],
        [0, 0.18421053, 0],
        [0, 0, 1],
    ]
)
_INV_SCALE = np.linalg.inv(_SCALE)

_SCALE = torch.from_numpy(_SCALE).to(_DEVICE).float()
_INV_SCALE = torch.from_numpy(_INV_SCALE).to(_DEVICE).float()

class HomographyLoss:
    def __init__(self, num_cameras):
        self.num_cameras = num_cameras

        self.homography_no_truck = copy.deepcopy(_HOMOGRAPHY_MATRICES["no_truck"])
        self.homography_white_truck = copy.deepcopy(_HOMOGRAPHY_MATRICES["white_truck"])

        for transf in self.homography_no_truck.keys():
            self.homography_no_truck[transf] = (
                torch.from_numpy(self.homography_no_truck[transf].copy()).to(_DEVICE).unsqueeze(0).float()
            )
        
        for transf in self.homography_white_truck.keys():
            self.homography_white_truck[transf] = (
                torch.from_numpy(self.homography_white_truck[transf].copy()).to(_DEVICE).unsqueeze(0).float()
            )

    def __call__(self, embeddings, labels):
        # assert 1 == 0, "Working on getting labels to work in the loss function."

        (batch_size, num_cameras, embedding_dim, space_dim) = embeddings.shape
        assert num_cameras == self.num_cameras, num_cameras
        assert space_dim == 3, space_dim

        assert batch_size == 1 and labels.shape == (1, ), f"Only labels of batch size 1 have been developed, got batch size {batch_size} and labels {labels.shape}"

        m = self.homography_no_truck if labels.item() == 0 else self.homography_white_truck

        if self.num_cameras == 3:
            z_c = embeddings[:, 0]
            z_l = embeddings[:, 1]
            z_r = embeddings[:, 2]

            loss = (
                torch.norm(
                    z_c
                    - torch.bmm(z_l, _SCALE @ m["left_to_center"] @ _INV_SCALE),
                    dim=(1, 2),
                )
                + torch.norm(
                    z_l
                    - torch.bmm(z_c, _SCALE @ m["center_to_left"] @ _INV_SCALE),
                    dim=(1, 2),
                )
                + torch.norm(
                    z_c
                    - torch.bmm(z_r, _SCALE @ m["center_to_right"] @ _INV_SCALE),
                    dim=(1, 2),
                )
                + torch.norm(
                    z_r
                    - torch.bmm(z_c, _SCALE @ m["right_to_center"] @ _INV_SCALE),
                    dim=(1, 2),
                )
            )

        elif self.num_cameras == 2:
            z_c = embeddings[:, 0]
            z_l = embeddings[:, 1]

            loss = torch.norm(
                z_c - torch.bmm(z_l, _SCALE @ m["left_to_center"] @ _INV_SCALE),
                dim=(1, 2),
            )
        else:
            raise NotImplementedError

        return loss.mean()


def get_loss_func(num_cameras):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
        cfg: experiment config.
    """
    return HomographyLoss(num_cameras)
