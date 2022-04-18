# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from multiprocessing.sharedctypes import Value
import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
import tqdm

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Stip(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg

        self.fps = 1  # Frame Rate of sampled clips
        self.sample_interval = 7  # Interval at which a new set of frames is sampled
        self.frame_count = cfg.DATA.NUM_FRAMES  # Number of frames per set of clips
        self.offset_update_rate = 500  # Interval at which offsets are updated

        self.data_path = "/mnt/disks/homography/STIP_dataset"  # Location at which frames and timestamp data are stored
        self.left_img_path = self.data_path + "/{:s}/L_camera"
        self.right_img_path = self.data_path + "/{:s}/R_camera"
        self.center_img_path = self.data_path + "/{:s}/C_camera"
        self.img_path = "/{:06d}.jpg"  # (Frame ID)

        self.data = self.generate_frame_correspondences(self.fps, self.frame_count)

        self.no_truck = [
            "0927-2017-downtown-ann-1",
            "0928-2017-downtown-ann-1",
            "0927-2017-downtown-ann-3",
            "0927-2017-downtown-ann-2",
            "ANN-hanh-1",
            "ANN-hanh-2",
            "ANN-conor-2",
            "ANN-conor-1",
        ]

        self.white_truck = [
            "downtown-palo-alto-6",
            "dt-palo-alto-3",
            "mountain-view-2",
            "dt-san-jose",
            "sf-soma-1",
            "dt-san-jose-2",
            "downtown-palo-alto-1",
            "mountain-view-1",
            "sf-soma-2",
            "downtown-palo-alto-2",
            "dt-san-jose-3",
            "mountain-view-4",
            "sf-financial-4",
            "dt-palo-alto-2",
            "mountain-view-3",
            "sf-southbound-5",
            "dt-san-jose-4",
            "dt-palo-alto-1",
            "downtown-palo-alto-4",
            "downtown-palo-alto-6",
        ]

    def generate_frame_correspondences(self, fps, frame_count):
        """Returns list of tuples of format (video, [[left frame IDs], [center frame IDs], [right frame IDs]])"""
        ret = []
        step = 20 // fps
        video_paths = sorted(os.listdir(self.data_path))[:18]
        for video in tqdm.tqdm(video_paths):
            # Loads in timestamp sync data to use to update offsets
            timestamps = np.genfromtxt(
                os.path.join(self.data_path, video, "timestamp_sync.txt"), delimiter=" "
            )
            ts_range = [int(timestamps[0, 2]), int(timestamps[-1, 2])]
            offset = [
                0,
                0,
            ]  # Left and Right Frame ID offsets in regard to the Center Frame ID
            data = []

            # Aggregates corresponding L, C, and R frame IDs into a list
            for frame in sorted(os.listdir(self.center_img_path.format(video))):
                f_number = int(frame.split(".")[0])
                # Periodically readjust offsets based on timestamp_sync data
                if (
                    f_number % self.offset_update_rate == 0
                    and ts_range[0] < f_number < ts_range[1]
                ):
                    if f_number in list(timestamps[:, 2]):
                        _, _, L, C, R = timestamps[
                            list(timestamps[:, 2]).index(f_number)
                        ]
                        offset = [int(L - C), int(R - C)]
                data += [[f_number + offset[0], f_number, f_number + offset[1]]]

            # Samples aggregated list based on desired fps, frame count, and sample interval
            data = np.array(data)
            for i in range(
                0, data.shape[0] - (step * frame_count), self.sample_interval
            ):
                ret += [(video, data[i : i + step * frame_count : step].T)]
        return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns dict containing name of video and tensor of expected shape: (num_cameras, num_channels, num_frames, W, H)
        Current Settings: (3, 32, 3, 224, 224)
        """
        video_name, frame_ids = self.data[idx]

        if video_name in self.no_truck:
            label_id = 0
        elif video_name in self.white_truck:
            label_id = 1
        else:
            raise ValueError(
                f"No homography matrix exists for videos in folder {video_name}."
            )

        # Loads in necessary frames based on frame IDs provided by self.data
        img_paths = [
            [
                self.left_img_path.format(video_name) + self.img_path.format(fid)
                for fid in frame_ids[0]
            ],
            [
                self.center_img_path.format(video_name) + self.img_path.format(fid)
                for fid in frame_ids[1]
            ],
            [
                self.right_img_path.format(video_name) + self.img_path.format(fid)
                for fid in frame_ids[2]
            ],
        ]

        cameras = []
        for camera in img_paths:
            frames = []
            for path in camera:
                img = Image.open(path).convert("RGB")
                assert img.size == (1936, 1216), img.size
                img = np.asarray(img.resize((224, 224)))
                img = img.transpose((2, 0, 1))
                frames += [img]
            cameras += [frames]

        cameras = np.array(cameras)

        if self.cfg.DATA.NUM_CAMERAS == 2:
            cameras = cameras[:2]

        cameras = torch.from_numpy(np.array(cameras))

        return cameras, label_id
