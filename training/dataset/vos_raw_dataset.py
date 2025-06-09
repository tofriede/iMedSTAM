# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
from iopath.common.file_io import g_pathmgr

from training.dataset.vos_segment_loader import (
    NPZSegmentLoader,
)

def mask2D_to_bbox(gt2D):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, 6, 1)[0]
    scale_y, scale_x = gt2D.shape
    bbox_shift_x = int(bbox_shift * scale_x/256)
    bbox_shift_y = int(bbox_shift * scale_y/256)
    #print(f'{bbox_shift_x=} {bbox_shift_y=} with orig {bbox_shift=}')
    x_min = max(0, x_min - bbox_shift_x)
    x_max = min(W-1, x_max + bbox_shift_x)
    y_min = max(0, y_min - bbox_shift_y)
    y_max = min(H-1, y_max + bbox_shift_y)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D):
    boxes = []
    z_indices, _, _ = np.where(gt3D > 0)
    z_indices = np.unique(z_indices)

    # Find indices where the difference between consecutive elements is greater than 1
    split_indices = np.where(np.diff(z_indices) > 1)[0] + 1

    # Split the array into contiguous chunks
    chunks = np.split(z_indices, split_indices)

    for chunk in chunks:
        b_dict = {}
        z_min, z_max = np.min(chunk), np.max(chunk)
        z_middle = chunk[len(chunk)//2]
        
        D, H, W = gt3D.shape
        b_dict['z_min'] = z_min
        b_dict['z_max'] = z_max
        b_dict['z_mid'] = z_middle

        gt_mid = gt3D[z_middle]

        box_2d = mask2D_to_bbox(gt_mid)
        x_min, y_min, x_max, y_max = box_2d
        b_dict['z_mid_x_min'] = x_min
        b_dict['z_mid_y_min'] = y_min
        b_dict['z_mid_x_max'] = x_max
        b_dict['z_mid_y_max'] = y_max

        assert z_min == max(0, z_min)
        assert z_max == min(D-1, z_max)
        boxes.append(b_dict)
    
    return boxes


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class NPZRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder=None,
        file_list_txt=None,
        excluded_file_list_txt=None,
        sample_rate=1,
        single_object_mode=True,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.single_object_mode = single_object_mode
        self.no_bbox_pattern = re.compile(r"(brats|vessel)", re.IGNORECASE)

        # read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f]
        else:
            subset = []
            for root, _, files in os.walk(self.img_folder):
                for f in files:
                    full_path = os.path.join(root, f)
                    relative_path = os.path.relpath(full_path, self.img_folder)
                    subset.append(relative_path)

        # read and process excluded files if provided
        if excluded_file_list_txt is not None:
            with g_pathmgr.open(excluded_file_list_txt, "r") as f:
                excluded_files = [line.strip() for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.npz_file_names = sorted(
            [npz_file_name for npz_file_name in subset if npz_file_name not in excluded_files]
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        npz_file_name = self.npz_file_names[idx]
        npz_file_path = os.path.join(self.img_folder, npz_file_name)
        npz = np.load(npz_file_path, allow_pickle=True)

        gts = npz['gts']
        D, _, _ = gts.shape
        unique_labs = np.unique(gts)[1:]
        objects = []
        for lab in unique_labs:
            gt = gts==lab
            boxes = mask3D_to_bbox(gt)
            objects.append({ 'id': int(lab), 'boxes': boxes })


        frames = []
        for fid in range(0, D, self.sample_rate):
            frames.append(VOSFrame(fid, image_path=(npz_file_path + "#" + str(fid))))
        video = VOSVideo(npz_file_name, idx, frames)

        segment_loader = NPZSegmentLoader(npz_file_path)

        no_bbox = bool(self.no_bbox_pattern.search(npz_file_name))

        return video, segment_loader, objects, npz_file_name, no_bbox

    def __len__(self):
        return len(self.npz_file_names)