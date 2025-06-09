import numpy as np

from template.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

# taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)