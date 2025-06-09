# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

MAX_RETRIES = 1000

@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]
    obj_boundaries: tuple


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()
    

class RandomObjectSampler(VOSSampler):
    def __init__(
        self,
        reverse_time_prob=0.5,
    ):
        self.num_frames = 0
        self.max_num_objects = 1
        self.reverse_time_prob = reverse_time_prob

    def sample(self, video, objects, no_bbox=False):

        for retry in range(MAX_RETRIES):
            obj_idx = random.randint(0, len(objects) - 1)
            obj = objects[obj_idx]
            box = random.choice(obj['boxes'])

            obj_start_frame = box["z_min"]
            obj_end_frame = box["z_max"]

            start_frame = box["z_min"]
            mid_frame = box["z_mid"]
            end_frame = box["z_max"]

            if no_bbox:
                extension = random.randint(0, 15)
                start_frame = max(0, start_frame - extension)
                end_frame = min(len(video.frames) - 1, end_frame + extension)

            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = video.frames[start_frame:mid_frame + 1]
                frames = frames[::-1]
            else:
                frames = video.frames[mid_frame:end_frame + 1]

            if len(frames) > 25:
                start_idx = random.randint(0, len(frames) - 25)
                frames = frames[start_idx:start_idx + 25]

            # Get first frame object ids
            visible_object_ids = [obj['id']]

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        return SampledFramesAndObjects(
            frames=frames,
            object_ids=visible_object_ids,
            obj_boundaries=(obj_start_frame, obj_end_frame)
        )
