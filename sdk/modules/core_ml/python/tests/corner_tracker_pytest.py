# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import torch

from metavision_core_ml.corner_detection.corner_tracker import CornerTracker


def pytestcase_add_new_corners():
    """
    Check the corner tracker ability to add new corners with id
    """
    corner_tracker = CornerTracker(time_tolerance=1000, distance_tolerance=2)

    x = torch.as_tensor([10, 20, 30])
    y = torch.as_tensor([10, 20, 30])
    ts = 1000
    corners = torch.stack([x, y, ts * torch.ones_like(x)], axis=1)
    corner_tracker.update_current_corners(corners)

    assert corner_tracker.current_corners.shape == torch.Size([3, 4])

    assert (corner_tracker.current_corners[:, 3] == torch.as_tensor([0, 1, 2])).all()


def pytestcase_match_corners_in_space():
    """
    Check the corner tracker ability to match corners in space
    """
    corner_tracker = CornerTracker(time_tolerance=1000, distance_tolerance=2)

    x = torch.as_tensor([10, 20, 30])
    y = torch.as_tensor([10, 20, 30])
    ts = 1000
    corners_a = torch.stack([x, y, ts * torch.ones_like(x)], axis=1)
    corner_tracker.update_current_corners(corners_a)
    x = torch.as_tensor([11, 20, 50])
    y = torch.as_tensor([10, 19, 30])
    corners_b = torch.stack([x, y, ts * torch.ones_like(x)], axis=1)
    corner_tracker.update_current_corners(corners_b)

    # Only one new corner is created
    assert corner_tracker.current_corners.shape == torch.Size([4, 4])

    # Check if others are updated.
    assert corner_tracker.current_corners[corner_tracker.current_corners[:, 1] == 10, 0] == torch.as_tensor([11])
    assert corner_tracker.current_corners[corner_tracker.current_corners[:, 0] == 20, 1] == torch.as_tensor([19])

    # New id created for unmatched corner
    assert corner_tracker.current_corners[corner_tracker.current_corners[:, 0] == 50, 3] == torch.as_tensor([3])


def pytestcase_match_corners_in_time():
    """
    Check the corner tracker ability to match corners in time
    """
    corner_tracker = CornerTracker(time_tolerance=1000, distance_tolerance=2)

    x = torch.as_tensor([10, 20, 30])
    y = torch.as_tensor([10, 20, 30])
    ts = 1000
    corners_a = torch.stack([x, y, ts * torch.ones_like(x)], axis=1)
    corner_tracker.update_current_corners(corners_a)
    ts = torch.as_tensor([1500, 2000, 3000])
    corners_b = torch.stack([x, y, ts], axis=1)
    corner_tracker.update_current_corners(corners_b)

    # Assert matched corners are updated
    assert corner_tracker.current_corners[corner_tracker.current_corners[:, 0] == 10, 2] != torch.as_tensor([1000])
    assert corner_tracker.current_corners[corner_tracker.current_corners[:, 0] == 20, 2] != torch.as_tensor([1000])

    # Assert new id is created for ts=3000
    assert corner_tracker.current_corners[corner_tracker.current_corners[:, 2] == 3000, 3] == torch.as_tensor([3])
