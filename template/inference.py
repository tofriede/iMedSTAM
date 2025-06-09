from typing import List, TypedDict, Optional
from functools import cmp_to_key

import torch
import numpy as np
import numpy.typing as npt

from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
from efficient_track_anything.efficienttam_video_predictor import EfficientTAMVideoPredictor

torch.manual_seed(42)
np.random.seed(42)

PREV_CASE_ID = None
PREV_STATE = None
CUR_ITERATION = 0
predictor = None

class Box(TypedDict):
    z_min: int
    z_max: int
    z_mid: int
    z_mid_x_min: int
    z_mid_x_max: int
    z_mid_y_min: int
    z_mid_y_max: int

class Clicks(TypedDict):
    fg: List[tuple[int, int, int]]
    bg: List[tuple[int, int, int]]

class SegmentationClass(TypedDict):
    id: int
    slices: tuple[int, int, int]


def get_model(
    config_path="configs/efficienttam/efficienttam_s_512x512.yaml",
    ckpt_path="./checkpoints/efficienttam_s_512x512.pt",
    device=torch.device("cuda")
) -> "EfficientTAMVideoPredictor":
    global predictor

    if predictor is None:
        predictor = build_efficienttam_video_predictor(
            config_file=config_path,
            ckpt_path=ckpt_path,
            device=device
        )

    return predictor

def get_relevant_slices(boxes: List[Box], depth: int) -> List[int]:
    lookups_left = dict()

    for box3D in boxes:
        z_min = max(0, box3D['z_min'])
        z_max = min(box3D['z_max'], depth)

        if z_min == z_max:
            lookups_left[z_min] = lookups_left.get(z_min,0) + 1

        for z in range(z_min, z_max + 1):
            lookups_left[z] = lookups_left.get(z,0) +1

    return lookups_left.keys()

def sort_classes(a: SegmentationClass, b: SegmentationClass) -> int:
    a_start, _, a_end = a['slices']
    b_start, _, b_end = b['slices']
    if a_start == b_start:
        return a_end - b_end
    else:
        return a_start - b_start

@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def infer_3d(
    case_id: str,
    model_config_path: str,
    model_ckpt_path: str,
    img_3D: npt.NDArray[np.uint8],
    clicks: List[Clicks],
    clicks_order: List[str],
    boxes: Optional[npt.NDArray[np.object_]] = None,
    prev_state: Optional[dict] = None,
) -> npt.NDArray[np.uint8]:
    global PREV_STATE
    global PREV_CASE_ID
    global CUR_ITERATION

    if prev_state:
        PREV_STATE = prev_state
        PREV_CASE_ID = prev_state.get("case_id", None) 
        CUR_ITERATION = prev_state.get("iteration", 0)

    D = img_3D.shape[0]
    if boxes is not None:
        z_indices = sorted(get_relevant_slices(list(boxes), D))
        z_indices = range(z_indices[0], z_indices[-1] + 1) # fill holes
    else:
        z_indices = list(range(D))
        boxes = [None] * len(clicks)

    image_slices = [img_3D[z, :, :] for z in z_indices]
    segs_3D = np.zeros(img_3D.shape, dtype=np.uint8)

    def frame_id_2_slice_id(frame_id: int):
        return frame_id + z_indices[0]
    
    def slice_id_2_frame_id(slice_id: int):
        return slice_id - z_indices[0]
    
    predictor = get_model(model_config_path, model_ckpt_path)
    
    if PREV_CASE_ID == case_id:
        state = PREV_STATE
        CUR_ITERATION += 1
        if 'images' not in state:
            predictor.set_images(state, image_slices)
    else:
        state = predictor.init_state(image_slices, offload_video_to_cpu=True, offload_state_to_cpu=True)
        CUR_ITERATION = 0

    classes: List[SegmentationClass] = []

    # background is class 0
    for class_id, box in enumerate(boxes, start=1):
        class_clicks = clicks[class_id - 1] if clicks else []
        if box is not None:
            start_frame = slice_id_2_frame_id(box["z_min"])
            mid_frame = slice_id_2_frame_id(box["z_mid"])
            end_frame = slice_id_2_frame_id(box["z_max"])
            if CUR_ITERATION == 0:
                state["object_boundaries"][class_id] = (start_frame, mid_frame, end_frame)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    clear_old_points=True,
                    frame_idx=mid_frame,
                    obj_id=class_id,
                    box=np.array([box["z_mid_x_min"], box["z_mid_y_min"], box["z_mid_x_max"], box["z_mid_y_max"]], dtype=np.float32)
                )
        else:
            initial_click = class_clicks["fg"][0]
            click_z, click_y, click_x = initial_click
            start_frame = slice_id_2_frame_id(z_indices[0])
            mid_frame = int(slice_id_2_frame_id(click_z))
            end_frame = slice_id_2_frame_id(D - 1)
            if CUR_ITERATION == 0:
                state["object_boundaries"][class_id] = (start_frame, mid_frame, end_frame)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    clear_old_points=True,
                    frame_idx=mid_frame,
                    obj_id=class_id,
                    points=[[click_x, click_y]],
                    labels=[1]
                )

        classes.append({
            'id': class_id,
            'slices': (start_frame, mid_frame, end_frame)
        })

        if CUR_ITERATION > 0:
            cls_clicks_order = list(clicks_order[class_id - 1])
            if not cls_clicks_order or cls_clicks_order[-1] is None:
                print(f"Skipping class {class_id} as already perfect")
                continue

            cur_click_type = cls_clicks_order[-1]
            click_idx = cls_clicks_order.count(cur_click_type) - 1
            new_click = class_clicks[cur_click_type][click_idx]
            click_z, click_y, click_x = new_click
            predictor.add_new_points_or_box(
                inference_state=state,
                clear_old_points=False,
                frame_idx=slice_id_2_frame_id(click_z),
                obj_id=class_id,
                points=[[click_x, click_y]],
                labels=[1 if cur_click_type == 'fg' else 0]
            )

    classes = sorted(classes, key=cmp_to_key(sort_classes))

    predictions: dict[int, dict[int, torch.Tensor]] = {}

    def process_prediction(
        frame_idx: int,
        class_id: int,
        class_idx: int,
        mask_logits: torch.Tensor
    ) -> None:
        frame_dict = predictions.get(frame_idx, {})
        frame_dict[class_id] = mask_logits
        predictions[frame_idx] = frame_dict

        # check if frame_idx appears in unprocessed classes
        is_needed = any(seg_class['slices'][0] <= frame_idx <= seg_class['slices'][2] for seg_class in classes[class_idx + 1:])
        if not is_needed:
            class_ids = list(frame_dict.keys())
            all_pred_masks = torch.cat(list(frame_dict.values()), dim=0)
            all_pred_masks = predictor.apply_non_overlapping_constraints(all_pred_masks)
            for i, class_id in enumerate(class_ids):
                mask_logits = all_pred_masks[i]
                segs_3D[frame_id_2_slice_id(frame_idx), (mask_logits > 0.0).cpu().numpy()] = class_id
            del predictions[frame_idx]
    
    for class_idx, seg_class in enumerate(classes):
        class_id = seg_class['id']
        start_frame, mid_frame, end_frame = seg_class['slices']

        for out_frame_id, _, out_mask_logits, _ in predictor.propagate_in_video(state, start_frame_idx=mid_frame, max_frame_num_to_track=(mid_frame - start_frame), obj_id=class_id, reverse=True):
            process_prediction(out_frame_id, class_id, class_idx, out_mask_logits[0])

        for out_frame_id, _, out_mask_logits, _ in predictor.propagate_in_video(state, start_frame_idx=mid_frame, max_frame_num_to_track=(end_frame - mid_frame), obj_id=class_id):
            if out_frame_id == mid_frame:
                continue # already processed in reverse propagation
            process_prediction(out_frame_id, class_id, class_idx, out_mask_logits[0])

    PREV_CASE_ID = case_id
    PREV_STATE = state

    state['case_id'] = case_id
    state['iteration'] = CUR_ITERATION
    state['allowed_running_time'] = len(boxes) * 90

    return segs_3D, state
