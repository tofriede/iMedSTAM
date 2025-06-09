# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100

class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a video
                video, segment_loader, objects, npz_file_name, no_bbox = self.video_dataset.get_video(idx)
                # sample frames and object indices to be used in a datapoint
                sampled_frms_and_objs = self.sampler.sample(
                    video, objects, no_bbox
                )
                datapoint = self.construct(video, sampled_frms_and_objs, segment_loader, npz_file_name, no_bbox)
                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)

                initial_frame_mask = datapoint.frames[0].objects[0].segment
                if initial_frame_mask.max() == 0:
                    # If the initial frame mask is empty, we need to retry
                    raise ValueError("Initial frame mask is empty")
                
                return datapoint
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e

    def construct(self, video, sampled_frms_and_objs, segment_loader, npz_file_name, no_bbox):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids
        obj_boundaries = sampled_frms_and_objs.obj_boundaries

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )
            hide = True if frame.frame_idx < obj_boundaries[0] or frame.frame_idx > obj_boundaries[1] else False
            segments = segment_loader.load(frame.frame_idx, sampled_object_ids[0], hide=hide)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
            file=npz_file_name,
            no_bbox=no_bbox
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}

    npz_file = None

    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            if ".npz" in path:
                npz_file_path = path.split("#")[0]
                frame_idx = int(path.split("#")[1])
                if npz_file is None:
                    npz_file = np.load(npz_file_path, allow_pickle=True)
                all_images.append(PILImage.fromarray(npz_file["imgs"][frame_idx]).convert("RGB"))
            else:
                with g_pathmgr.open(path, "rb") as fopen:
                    all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)
