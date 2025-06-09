import numpy as np

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
    x_min = max(0, x_min - bbox_shift_x)
    x_max = min(W-1, x_max + bbox_shift_x)
    y_min = max(0, y_min - bbox_shift_y)
    y_max = min(H-1, y_max + bbox_shift_y)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D):
    b_dict = {}
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    z_indices = np.unique(z_indices)
    # middle of z_indices
    z_middle = z_indices[len(z_indices)//2]
    
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
    return b_dict