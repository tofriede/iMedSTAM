# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import torch.distributed

from efficient_track_anything.modeling.efficienttam_base import EfficientTAMBase
from efficient_track_anything.modeling.efficienttam_utils import (
    sample_box_points,
    sample_box_points_old,
    sample_initial_points,
)
from efficient_track_anything.utils.misc import concat_points
from training.utils.data_utils import BatchedVideoDatapoint
from template.error import get_refinement_click_from_largest_error


class EfficientTAMTrain(EfficientTAMBase):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=True,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input: BatchedVideoDatapoint):
        device = input.masks.device
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        backbone_out["correction_pt_frames"] = []
        output_dict = self.forward_tracking(backbone_out, input)

        gt = torch.squeeze(input.masks, dim=1).cpu().numpy().astype(np.uint8) # remove object dimension (always 1)
        all_frame_outputs = self._merge_output_dict(output_dict, input.num_frames)

        for i in range(3):
            segs_3D = np.zeros(gt.shape, dtype=np.uint8)
            assert len(all_frame_outputs) == gt.shape[0]

            for frame_idx, output in enumerate(all_frame_outputs):
                pred_masks = output["pred_masks_high_res"]
                pred_mask = pred_masks[0][0]
                assert pred_mask.shape[0] == gt.shape[1]
                assert pred_mask.shape[1] == gt.shape[2]
                segs_3D[frame_idx, (pred_mask > 0.0).cpu().numpy()] = 1

            click = get_refinement_click_from_largest_error(segs_3D, gt)
            if click is None:
                continue

            click_z, click_y, click_x = click["values"]
            slice = int(click_z)
            if slice not in backbone_out["correction_pt_frames"]:
                backbone_out["correction_pt_frames"].append(slice)

            new_points=torch.tensor([[[click_x, click_y]]], dtype=torch.int, device=device)
            new_labels=torch.tensor([[1 if click["type"] == 'fg' else 0]], dtype=torch.int, device=device)
            point_inputs = backbone_out["point_inputs_per_frame"].get(slice, None)
            if point_inputs is None:
                point_inputs = { "point_coords": new_points, "point_labels": new_labels }
            else:
                point_inputs = concat_points(point_inputs, new_points, new_labels)
            backbone_out["point_inputs_per_frame"][slice] = point_inputs

            output_dict = self.forward_tracking(backbone_out, input, output_dict)
            all_frame_outputs = self._merge_output_dict(output_dict, input.num_frames)

        return all_frame_outputs
    
    def _merge_output_dict(self, output_dict, num_frames: int):
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]

        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input: BatchedVideoDatapoint, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames
        backbone_out["use_pt_input"] = False

        init_cond_frames = [start_frame_idx]  # starting frame
        backbone_out["init_cond_frames"] = [start_frame_idx]
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if input.no_bbox[0]:
                points, labels = sample_initial_points(
                    gt_masks_per_frame[t],
                    input.masks,
                    input.files[0]
                )
            else:
                try:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                except Exception as e:
                    points, labels = sample_box_points_old(
                        gt_masks_per_frame[t],
                    )
            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][t] = point_inputs

        return backbone_out

    def forward_tracking(
        self,
        backbone_out,
        input: BatchedVideoDatapoint,
        output_dict=None
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        correction_pt_frames = sorted(backbone_out["correction_pt_frames"])
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        remaining_correction_pt_frames = [frame_idx for frame_idx in correction_pt_frames if frame_idx not in init_cond_frames]  
        remaining_frames = [frame_idx for frame_idx in backbone_out["frames_not_in_init_cond"] if frame_idx not in remaining_correction_pt_frames]
        processing_order = init_cond_frames + remaining_correction_pt_frames + remaining_frames
        if output_dict is None:
            output_dict = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            prev_sam_mask_logits = None
            if stage_id in correction_pt_frames:
                prev_out = output_dict["cond_frame_outputs"].get(stage_id)
                if prev_out is None:
                    prev_out = output_dict["non_cond_frame_outputs"].get(stage_id)

                if prev_out is not None and prev_out["pred_masks"] is not None:
                    prev_sam_mask_logits = prev_out["pred_masks"]

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                output_dict=output_dict,
                prev_sam_mask_logits=prev_sam_mask_logits,
                num_frames=num_frames,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in correction_pt_frames
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
                if stage_id in output_dict["non_cond_frame_outputs"]:
                    del output_dict["non_cond_frame_outputs"][stage_id]
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        return output_dict

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        gt_masks=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            None,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs


        prev_out = output_dict["cond_frame_outputs"].get(frame_idx)
        if prev_out is None:
            prev_out = output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is None:
            current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
            current_out["multistep_pred_ious"] = [ious]
            current_out["multistep_object_score_logits"] = [object_score_logits]
        else: 
            multistep_keys = [
                "multistep_pred_multimasks_high_res",
                "multistep_pred_ious",
                "multistep_object_score_logits"
            ]
            for key in multistep_keys:
                current_out[key] = prev_out[key]

            current_out["multistep_pred_multimasks_high_res"].append(high_res_multimasks)
            current_out["multistep_pred_ious"].append(ious)
            current_out["multistep_object_score_logits"].append(object_score_logits)

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out