# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

class NPZSegmentLoader:
    def __init__(self, npz_file_path):
        npz_file = np.load(npz_file_path, allow_pickle=True)
        self.gts = npz_file["gts"]

    def load(self, frame_id, obj_id, hide=False):
        # load the mask
        binary_segments = {}
        if hide:
              binary_segments[obj_id] = torch.full(self.gts[frame_id].shape, False)
        else:
            binary_segments[obj_id] = torch.from_numpy(self.gts[frame_id] == obj_id)
        return binary_segments

    def __len__(self):
        return