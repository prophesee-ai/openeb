# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
class to create tracks from corners based on distance in space and time
"""

import numpy as np
import cv2
import torch

from collections import defaultdict


class CornerTracker:
    def __init__(self, time_tolerance, distance_tolerance=3):
        """
        time tolerance: (int) time in us
        distance tolerance: (int) distance in pixels
        """
        self.time_tolerance = time_tolerance
        self.distance_tolerance = distance_tolerance

        self.current_corners = torch.zeros((0, 4))
        self.current_track_id = 0

        # for vis
        self.traj = defaultdict(list)
        self.updates = defaultdict(int)
        self.colormap = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)[:, 0]
        self.max_diff_t = 10000

        # count track lengths
        self.track_len = 0
        self.num_tracks = 0

    def add_track_id_to_event_buffer(self, events, track_id=None):
        """
        Concatenates events with track ids. If track ids are given use them otherwise create new ids
        Args:
            events: torch tensor of shape Nx3 (x, y, t)
            track_id: torch tensor of shape N or Nx1

        Returns:
            Events and ids of shape Nx4
        """
        if track_id is None:
            current_corners = torch.cat(
                [events, torch.arange(self.current_track_id, self.current_track_id + len(events)).view(-1, 1)], axis=1)
            self.current_track_id += len(events)
        else:
            current_corners = torch.cat([events, track_id], axis=1)
        return current_corners

    def match_in_space_corners(self, new_corners, old_corners):
        """
        Match two set of corners in space, i.e. only considers x, y position for nearest neighbor matching
        Args:
            new_corners: torch tensor of shape (N, 3) comprised of x, y, t
            old_corners: torch tensor of shape (N, 3) comprised of x, y, t

        Returns:
            index_matched_new: index of corners in new_corners array matched
            index_unmatched_new: index of corners in new_corners array not matched
            index_matched_old: index of corners in old_corners array matched corresponding to index_matched_new
            index_unmatched_old: index of corners in old_corners array not matched
        """
        new_corners_coord = new_corners[:, :2]
        old_corners_coord = old_corners[:, :2]

        dist = torch.cdist(new_corners_coord.float(), old_corners_coord.float())
        dist_value, index_old_corners = dist.min(1)
        index_new_corners = torch.arange(len(new_corners))
        dist_value_mask = dist_value <= self.distance_tolerance
        index_matched_new = index_new_corners[dist_value_mask]
        index_unmatched_new = index_new_corners[~dist_value_mask]
        index_matched_old = index_old_corners[dist_value_mask]
        index_unmatched_old = index_old_corners[~dist_value_mask]
        return index_matched_new, index_unmatched_new, index_matched_old, index_unmatched_old

    def match_in_time_corners(self, new_corners, old_corners):
        """
        Match two set of corners in time, i.e. only considers time stamps for nearest neighbor matching
        Args:
            new_corners: torch tensor of shape (N, 3) comprised of x, y, t
            old_corners: torch tensor of shape (N, 3) comprised of x, y, t

        Returns:
            index of corners in new_corners matched
            index of corners in new_corners not matched

        """
        index = torch.arange(len(new_corners))
        mask = new_corners[:, 2] - old_corners[:, 2] <= self.time_tolerance
        return index[mask], index[~mask]

    def add_new_corners(self, new_corners):
        """
        Adds new corners to the tracker updating its internal state and adding ids to corners.
        This is the main function to call over time.
        Args:
            new_corners: Torch tensor of size Nx3 corresponding to N new corners
        """
        # Positional matching
        index_matched_new, index_unmatched_new, index_matched_old, index_unmatched_old = self.match_in_space_corners(
            new_corners, self.current_corners)
        new_matched_corners = new_corners[index_matched_new]
        new_unmatched_corners = new_corners[index_unmatched_new]
        old_matched_corners = self.current_corners[index_matched_old]
        self.current_corners = self.current_corners[index_unmatched_old]

        # temporal matching
        index_temporal_matched, index_temporal_unmatched = self.match_in_time_corners(new_matched_corners,
                                                                                      old_matched_corners)

        new_unmatched_corners = torch.cat([new_unmatched_corners, new_matched_corners[index_temporal_unmatched]])
        new_matched_corners = new_matched_corners[index_temporal_matched]
        new_matched_ids = old_matched_corners[index_temporal_matched, 3]

        # adding the id of matched points
        new_matched_corners = self.add_track_id_to_event_buffer(new_matched_corners, new_matched_ids.view(-1, 1))
        new_unmatched_corners = self.add_track_id_to_event_buffer(new_unmatched_corners)

        # new current corners are composed of (not too) old unmatched corners and new corners
        self.current_corners = torch.cat([self.current_corners, new_matched_corners, new_unmatched_corners])

    def remove_old_events(self, ts):
        self.current_corners = self.current_corners[ts - self.current_corners[:, 2] <= self.time_tolerance]

    def update_current_corners(self, events):
        if len(events) != 0:
            self.remove_old_events(events[:, 2].min())
            if len(self.current_corners) == 0:
                self.current_corners = self.add_track_id_to_event_buffer(events)
            else:
                self.add_new_corners(events)

    def show(self, frame):
        for x, y, ts, id_ in self.current_corners:
            self.traj[int(id_)].append((int(x), int(y)))
            self.updates[int(id_)] = int(ts)

        invalid_ids = []
        for idx, t in self.updates.items():
            diff_t = ts - t
            if diff_t >= self.max_diff_t:
                invalid_ids.append(idx)
        for idx in invalid_ids:
            del self.traj[idx]
            del self.updates[idx]

        # draw on frame the trajectories
        frame = frame.copy()
        for idx, trajectory in self.traj.items():
            color = self.colormap[idx % 255].tolist()
            cv2.circle(frame, trajectory[0], 0, color, 5)
            self.track_len += len(trajectory)
            self.num_tracks += 1
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i - 1]
                pt2 = trajectory[i]
                cv2.line(frame, pt1, pt2, color, 1)
        return frame
