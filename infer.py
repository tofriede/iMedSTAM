import os
import argparse
import glob
import time

import numpy as np
import torch

from template.inference import infer_3d

DOCKER_OVERHEAD = 4

def check_skip_next_iterations(state, remaining_iterations: int) -> bool:
    prev_iterations = state.get("iteration", 0) + 1
    real_running_time = state['running_time']
    avg_time_per_iter = real_running_time / prev_iterations
    remaining_time = state['allowed_running_time'] - real_running_time
    needed_time = avg_time_per_iter + (remaining_iterations - 1) * (DOCKER_OVERHEAD + 1)
    return remaining_time < needed_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_path", type=str, default="./tests")
    parser.add_argument("--save_path", type=str, default="./outputs/")
    parser.add_argument("--model", type=str, default="checkpoints/efficienttam_s_512x512.pt")
    parser.add_argument("--model_config", type=str, default="configs/efficienttam/efficienttam_s_512x512_no_compile.yaml")
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    tmp_state_path = os.path.join(args.save_path, "tmp_model_state.pt")
    skip_path = os.path.join(args.save_path, "skip.txt")

    # load data
    test_cases = glob.glob(os.path.join(args.test_img_path, "*.npz"))
    for img_path in test_cases:
        case_name = os.path.basename(img_path)

        if os.path.exists(skip_path):
            with open(skip_path, "r") as f:
                skip_case_name = f.read().strip()
                if skip_case_name == case_name:
                    print(f"Skipping inference for case {case_name} as remaining time is less than average time per iteration")
                    continue
                else:
                    os.remove(skip_path)

        start_time = time.time()

        input_img = np.load(img_path, allow_pickle=True)
        no_bbox = False if "boxes" in input_img.keys() else True
        first_iteration = False if "prev_pred" in input_img.keys() else True

        prev_state = None
        if not first_iteration and os.path.exists(tmp_state_path):
            tmp_state = torch.load(tmp_state_path, weights_only=False)
            prev_iteration = tmp_state.get("iteration", 0)
            prev_case_id = tmp_state.get("case_id", None)

            if prev_case_id == case_name:
                print(f"Loading state from {tmp_state_path} for case {prev_case_id} with iteration {prev_iteration}")
                prev_state = tmp_state
              

        segs, state = infer_3d(
            case_id=case_name,
            model_config_path=args.model_config,
            model_ckpt_path=args.model,
            img_3D=input_img['imgs'],
            clicks=list(input_img['clicks']) if "clicks" in input_img.keys() else [],
            clicks_order=list(input_img['clicks_order']) if "clicks_order" in input_img.keys() else [],
            boxes=input_img['boxes'] if "boxes" in input_img.keys() else None,
            prev_state=prev_state
        )

        np.savez_compressed(
            os.path.join(args.save_path, case_name),
            segs=segs
        )

        current_iteration = state.get("iteration", 0)
        total_iterations = 5 if no_bbox else 6
        remaining_iterations = total_iterations - (current_iteration + 1)
        if remaining_iterations > 0:
            state['running_time'] = state.get('running_time', 0) + (time.time() - start_time) + DOCKER_OVERHEAD
            skip_next_iterations = check_skip_next_iterations(state, remaining_iterations)
            if skip_next_iterations:
                with open(skip_path, "w") as f:
                    f.write(case_name)
            else:
                del state['images']
                torch.save(state, tmp_state_path)