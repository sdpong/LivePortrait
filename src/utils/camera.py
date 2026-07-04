# coding: utf-8

"""
functions for processing and transforming 3D facial keypoints
"""

import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi


# Cache for the index tensor used in headpose_pred_to_degree.
# Avoids creating a new FloatTensor + device transfer on every call.
_idx_tensor_cache = {}


def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    
    Optimization: the idx_tensor (0..65) is cached per (device, dtype) 
    to avoid creating it on CPU and transferring to device every call.
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        cache_key = (pred.device, pred.dtype)
        if cache_key not in _idx_tensor_cache:
            _idx_tensor_cache[cache_key] = torch.arange(66, dtype=pred.dtype, device=pred.device)
        idx_tensor = _idx_tensor_cache[cache_key]
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred * idx_tensor, dim=1) * 3 - 97.5

        return degree

    return pred


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    
    Optimization: creates tensors directly on the target device instead of
    on CPU + .to(device). Also uses torch.stack instead of torch.cat+reshape
    which is slightly more efficient for matrix construction.
    """
    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1], device=device)
    zeros = torch.zeros([bs, 1], device=device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.stack([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.stack([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.stack([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose
