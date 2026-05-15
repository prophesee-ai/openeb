# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Help function for corner detection
"""
import numpy as np
from scipy import ndimage
import cv2
from numba import njit

import torch

from metavision_core_ml.core.temporal_modules import time_to_batch, batch_to_time


@njit
def numpy_nms(input_array, size=7):
    """
        runs non maximal suppression on square patches of size x size on the two last dimension
        Args:
            input_tensor: numpy array of shape B, C, H, W
            size (int): size of the side of the square patch for NMS

        Returns:
            numpy array where local maximas are unchanged and all other values are -10e5
        """
    B, C, H, W = input_array.shape
    output = np.ones((B, C, H, W)) * float("-inf")
    for b in range(B):
        for c in range(C):
            for y in range(0, H - size + 1, size):
                for x in range(0, W - size + 1, size):
                    patch = input_array[b, c, y:y + size, x:x + size]
                    index = np.argmax(patch)
                    index_x = index % size
                    index_y = index // size
                    output[b, c, y + index_y, x + index_x] = input_array[b, c, y + index_y, x + index_x]
    return output


def torch_nms(input_tensor, kernel_size=7):
    """
    runs non maximal suppression on square patches of size x size on the two last dimension
    Args:
        input_tensor: torch tensor of shape B, C, H, W
        kernel_size (int): size of the side of the square patch for NMS

    Returns:
        torch tensor where local maximas are unchanged and all other values are -inf
    """
    B, C, H, W = input_tensor.shape
    val, idx = torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, return_indices=True)
    offsets = torch.arange(B * C, device=input_tensor.device) * H * W
    offsets = offsets.repeat_interleave(H // kernel_size).repeat_interleave(W // kernel_size).reshape(B, C,
                                                                                                      H // kernel_size,
                                                                                                      W // kernel_size)
    output_tensor = torch.ones_like(input_tensor) * float("-inf")
    output_tensor.view(-1)[idx + offsets] = val

    return output_tensor


def get_harris_corners_from_image(img, return_mask=False):
    """
        takes an image as input and outputs harris corners

        Args:
            img: opencv image
            return_mask: returns a binary heatmap instead of corners positions
        Returns:
            harris corners in 3d with constant depth of one or a binary heatmap
    """
    harris_corners = cv2.cornerHarris(img, 5, 3, 0.06)
    filtered = ndimage.maximum_filter(harris_corners, 7)
    mask = (harris_corners == filtered)
    harris_corners *= mask
    mask = harris_corners > 0.001
    if mask.sum() == 0:
        mask = harris_corners >= 0.5 * harris_corners.max()
    mask = 1 * mask
    if return_mask:
        return mask
    y, x = np.nonzero(mask)
    harris_corners = np.ones((len(x), 3))
    harris_corners[:, 0] = x
    harris_corners[:, 1] = y
    return harris_corners


def project_points(points,
                   homography,
                   width,
                   height,
                   original_width,
                   original_height,
                   return_z=False,
                   return_mask=False,
                   filter_correct_corners=True):
    """
        projects 2d points given an homography and resize new points to new dimension.

        Args:
            points: 2d points in the form [x, y, 1] numpy array shape Nx3
            homography: 3*3 homography numpy array
            width: desired new dimension
            height: desired new dimension
            original_width: original dimension in which homography is given
            original_height: original dimension in which homography is given
            return_z: boolean to return points as 2d or 3d
            return_mask: boolean to return mask of out-of-bounds projected points
            filter_correct_corners: boolean whether to filter out-of-bounds projected points or not


        Returns:
            projected points: points projected in the new space and filtered by default to output only correct points
    """
    projected_points = np.matmul(homography, points.T).T
    projected_points /= np.expand_dims(projected_points[:, 2], 1)
    if return_z:
        projected_points = projected_points[:, :3]
    else:
        projected_points = projected_points[:, :2]
    projected_points[:, 0] *= width / original_width
    projected_points[:, 1] *= height / original_height
    if not filter_correct_corners and not return_mask:
        return projected_points
    mask = (projected_points[:, 0].round() >= 0) * (projected_points[:, 1].round() >= 0) * \
           (projected_points[:, 0].round() < width) * (projected_points[:, 1].round() < height)
    if not filter_correct_corners:
        return projected_points, mask
    else:
        if return_mask:
            return projected_points[mask], mask
        else:
            return projected_points[mask]


def update_ccl_tracker(tracker, y, x, events_dtype, ts):
    """
    Update ccl tracker from torch tensors
    Args:
        tracker: ccl tracker class instance
        y: torch tensor of corners y positions
        x: torch tensor of corners x positions
        events_dtype: dtype of events
        ts: timestamp of corners

    Returns:
        updated tracker instance
    """
    n = len(y)
    events_buf = np.zeros((n,), dtype=events_dtype)
    events_buf[:n]['y'] = y.cpu().numpy()
    events_buf[:n]['t'] = ts
    events_buf[:n]['x'] = x.cpu().numpy()
    events_buf[:n]['p'] = 1
    tracker(events_buf)
    return tracker


def update_nn_tracker(tracker, x, y, ts):
    """
    Update nearest neighbors tracker from torch tensors
    Args:
        tracker: nearest neighbor tracker class instance
        y: torch tensor of corners y positions
        x: torch tensor of corners x positions
        ts: timestamp of corners

    Returns:
        updated tracker instance
    """
    torch_corners = torch.stack([x,
                                 y,
                                 ts * torch.ones_like(x)],
                                axis=1).cpu()
    tracker.update_current_corners(torch_corners)
    return tracker


def save_ccl_corners(tracker, csv_writer, ts):
    """
    Extract corners from the tracker and writes them to a csv
    Args:
        tracker: ccl tracker class instance
        csv_writer: csv writer
        ts: timestamp of corners

    """
    corners = tracker.flow_buffer.numpy()
    ids = np.unique(corners['id']).tolist()
    for id in ids:
        i = np.where(corners['id'] == id)[0][0]
        val = corners[i]
        cx, cy = int(val['center_x']), int(val['center_y'])
        csv_writer.writerow([cx, cy, ts, id])


def save_nn_corners(tracker, csv_writer, ts):
    """
    Extract corners from the tracker and writes them to a csv
    Args:
        tracker: nearest neighbors tracker class instance
        csv_writer: csv writer
        ts: timestamp of corners

    """
    corners = tracker.current_corners.int().numpy()
    for (x, y, t, id) in corners:
        csv_writer.writerow([x, y, ts, id])


def clean_pred(pred, threshold=0.3):
    """
    Create a binary mask from a prediction between 0 and 1 after removal of local maximas
    Args:
        pred: prediction of the network after the sigmoid layer TxBxCxHxW
        threshold: Value of local maximas to consider corners

    Returns:
    Binary mask of corners locations.
    """
    pred, batch_size = time_to_batch(pred)
    pred = torch_nms(pred)
    if pred.is_cuda:
        threshold = torch.tensor(threshold).to(pred.get_device())
    else:
        threshold = torch.as_tensor(threshold)
    pred_thresholded = 1 * (pred > threshold.view(-1, 1, 1, 1))
    pred = batch_to_time(pred_thresholded, batch_size)
    return pred


@njit
def events_as_pol(events, frame):
    """
    From events creates an image
    Args:
        events: events psee format
        frame: numpy array to show events on

    Returns:
        updated frame
    """
    for i in range(events.shape[0]):
        x, y, p, t = events["x"][i], events["y"][i], events["p"][i], events["t"][i]
        frame[y, x, :] = p*255
    return frame
