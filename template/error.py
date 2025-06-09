from typing import TypedDict, Literal

import numpy as np
import torch
import cc3d
from scipy.ndimage import distance_transform_edt 

class RefinementClick(TypedDict):
    type: Literal['bg', 'fg']
    values: list[int]

def sample_coord(edt):
    # Find all coordinates with max EDT value
    np.random.seed(42)

    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)

    # Uniformly choose one of them
    chosen_index = max_coords[np.random.choice(len(max_coords))]

    center = tuple(chosen_index)
    return center

# Compute the EDT with same shape as the image
def compute_edt(error_component):
    # Get bounding box of the largest error component to limit computation
    coords = np.argwhere(error_component)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    crop_shape = max_coords - min_coords

    # Compute padding (25% of crop size in each dimension)
    padding =  np.maximum((crop_shape * 0.25).astype(int), 1)


    # Define new padded shape
    padded_shape = crop_shape + 2 * padding

    # Create new empty array with padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)

    # Fill center region with actual cropped data
    center_crop[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ] = error_component[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    large_roi = False
    if center_crop.shape[0] * center_crop.shape[1] * center_crop.shape[2] > 60000000:
        from skimage.measure import block_reduce
        print(f'ROI too large {center_crop.shape} --> 2x downsampling for EDT')
        center_crop = block_reduce(center_crop, block_size=(2, 2, 2), func=np.max)
        large_roi = True

    # Compute EDT on the padded array
    if torch.cuda.is_available() and not large_roi: # GPU available
        import cupy as cp
        from cucim.core.operations import morphology
        error_mask_cp = cp.array(center_crop)
        edt_cp = morphology.distance_transform_edt(error_mask_cp, return_distances=True)
        edt = cp.asnumpy(edt_cp)
    else: # CPU available only
        edt = distance_transform_edt(center_crop)
    
    if large_roi: # upsample
        edt = edt.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)

    # Crop out the center (remove padding)
    dist_cropped = edt[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ]

    # Create full-sized EDT result array and splat back 
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ] = dist_cropped

    dist_transformed = dist_full

    return dist_transformed

def get_refinement_click_from_largest_error(
    segs_cls: np.ndarray,
    gts_cls: np.ndarray,
) -> RefinementClick | None:
    # Compute error mask
    error_mask = (segs_cls != gts_cls).astype(np.uint8)
    if np.sum(error_mask) > 0:
        errors = cc3d.connected_components(error_mask, connectivity=26)  # 26 for 3D connectivity

        # Calculate the sizes of connected error components
        component_sizes = np.bincount(errors.flat)

        # Ignore non-error regions 
        component_sizes[0] = 0

        # Find the largest error component
        largest_component_error = np.argmax(component_sizes)

        # Find the voxel coordinates of the largest error component
        largest_component = (errors == largest_component_error)

        edt = compute_edt(largest_component)
        edt *= largest_component # make sure correct voxels have a distance of 0
        if np.sum(edt) == 0: # no valid voxels to sample
            edt = largest_component # in case EDT is empty (due to artifacts in resizing, simply sample a random voxel from the component), happens only for extremely small errors

        center = sample_coord(edt)
        assert largest_component[center] # click within error

        if gts_cls[center] == 0: # oversegmentation -> place background click
            assert segs_cls[center] == 1
            return { 'type': 'bg', 'values': list(center)}
        else: # undersegmentation -> place foreground click
            assert segs_cls[center] == 0
            return { 'type': 'fg', 'values': list(center)}

    else:
        return None