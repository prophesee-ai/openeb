# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
functions to compute homography reprojection error from
Manderscheid, J., Sironi, A., Bourdis, N., Migliore, D., & Lepetit, V.
Speed invariant time surface for learning to detect corner points with event-based cameras.
CVPR 2019.
"""
import argparse
import os
import numpy as np
import cv2
import torch
import csv

from metavision_core.event_io.py_reader import EventNpyReader

FEATURE_TYPE = {'names': ['x', 'y', 'id', 't'], 'formats': ['<u2', '<u2', '<i4', '<i8'],
                'offsets': [0, 2, 4, 8], 'itemsize': 16}


def convert_csv_to_npy(csv_path, overwrite=False):
    """
    Converts csv files to npy enable use of EventNpyReader class
    Args:
        csv_path: path of csv
        overwrite: if csv exists overwrite or not
    """
    npy_path = os.path.join(csv_path.replace(".csv", "corners_features.npy"))
    if os.path.exists(npy_path) and not overwrite:
        print("{} exists, not overwriting it".format(npy_path))
    else:
        with open(csv_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            # count_lines
            num_lines = 0
            next(csv_reader, None)  # skip header
            for _ in csv_reader:
                num_lines += 1
            buffer = np.zeros((num_lines,), dtype=FEATURE_TYPE)
            csvfile.seek(0)
            next(csv_reader, None)  # skip header
            for i, line in enumerate(csv_reader):
                x, y, t, id = line
                buffer[i]["x"] = x
                buffer[i]["y"] = y
                buffer[i]["t"] = t
                buffer[i]["id"] = id
        np.save(npy_path, buffer)


class ComputeKPI:
    """
    Computes reprojection error for planar scenes (eg: Atis dataset)
    """

    def __init__(self, npy_folder_path):
        """

        Args:
            npy_folder_path: npy_folder_path is the path of the folder containing a .npy file for each sequence to
            evaluate. Each file is a list of corners of type:
                                        {'names': ['x', 'y', 'id', 't'],
                                        'formats': ['<u2', '<u2', '<i4', '<i8'],
                                        'offsets': [0, 2, 4, 8], 'itemsize': 16}
            .npy files can be easily created using the function convert_csv_to_npy above.
        """
        self.npy_file_list = [os.path.join(npy_folder_path, path)
                              for path in os.listdir(npy_folder_path) if ".npy" in path]
        self.delta_t_list = [25000, 50000, 100000, 150000, 200000]
        self.window_size = 5000

    def compute_reprojection_error(self, old_corners, new_corners):
        src_points = []
        dst_points = []
        for id in np.unique(old_corners["id"]):
            if id in new_corners["id"]:
                old_corner = old_corners[old_corners["id"] == id][-1]
                new_corner = new_corners[new_corners["id"] == id][-1]
                src_points.append([int(old_corner["x"]), int(old_corner["y"])])
                dst_points.append([int(new_corner["x"]), int(new_corner["y"])])
        return homography_error(np.array(src_points), np.array(dst_points))

    def compute(self):
        # get reprojection error and print results
        reprojection_error_results = {}
        for delta_t in self.delta_t_list:
            results_per_delta_t = []
            for file_path in self.npy_file_list:
                npy_reader = EventNpyReader(file_path)
                npy_reader.seek_event(1)
                old_corners = npy_reader.load_delta_t(self.window_size)
                while not npy_reader.done:
                    ignore_corners = npy_reader.load_delta_t(delta_t - self.window_size)
                    new_corners = npy_reader.load_delta_t(self.window_size)
                    results_per_delta_t.append(self.compute_reprojection_error(old_corners, new_corners))
                    old_corners = new_corners

            results_per_delta_t = np.array(results_per_delta_t)
            num_matched_features = np.mean(results_per_delta_t[:, 0])
            reproj_error = np.mean(results_per_delta_t[results_per_delta_t[:, 0] != 0, 1])
            ratio_failed_matches = np.mean(results_per_delta_t[:, 0] == 0)
            reprojection_error_results[delta_t] = {"num_matched_features": num_matched_features,
                                                   "reprojection_error": reproj_error,
                                                   "ratio_failed_matches": ratio_failed_matches}
            print("delta_t: {} , num_matched_features: {} , reprojection_error: {} , ratio_failed_matches: {}".format(
                delta_t, num_matched_features, reproj_error, ratio_failed_matches
            ))

        # get longest track average
        track_length_list = []
        for file_path in self.npy_file_list:
            events = np.load(file_path)
            ids, counts = np.unique(events["id"], return_counts=True)
            index_top_hundred = np.argpartition(counts, -100)[-100:]
            total_track_length = 0
            for index, id in enumerate(ids[index_top_hundred]):
                ts = events[events["id"] == id]["t"]
                total_track_length += (ts[-1] - ts[0])
            total_track_length /= 100.
            track_length_list.append(total_track_length*10e-6)
        mean_track_length = np.array(track_length_list).mean()
        reprojection_error_results["Track length"] = mean_track_length
        print("average longest track length: {} seconds".format(mean_track_length))

        return reprojection_error_results


def homography_error(src_pts, dst_pts):
    """
    compute reprojection error between two set of corresponding points by estimating an homography between them
    Args:
        src_pts: numpy array of 2d corners coordinates in source scene (Nx2)
        dst_pts: numpy array of 2d corners coordinates corresponding supposedly to src_pts in destination scene (Nx2)

    Returns:
        The number of points matched, the reprojection error
        [0, -1] if no homography was found

    """
    if len(dst_pts) != 0 and len(src_pts) != 0:
        # Get only correspondences
        if len(src_pts) >= 6 and len(dst_pts) >= 6:
            mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mat is not None:
                mat = torch.tensor(mat)
                projection = project_coordinates(torch.tensor(src_pts).double(), mat)
                error = torch.sqrt(((projection - dst_pts) ** 2).sum(1))

                return [float(len(src_pts)), error.mean().cpu().numpy()]
    return [0, -1]


def compute_distance(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def project_coordinates(coordinates, homography):
    """
    Using the homography, projects coordinates
    Args:
        coordinates: 2d numpy array of shape Nx2
        homography: 3x3 homography matrix

    Returns:
        Projected and normalized coordinates
    """
    length_array = coordinates.shape[0]
    coordinates_extended = torch.cat((coordinates, torch.ones(
        (length_array, 1), dtype=torch.float64).to(coordinates.device)), 1)
    coordinates_projected = torch.mm(homography, coordinates_extended.T).T
    coordinates_projected_normalised = coordinates_projected / coordinates_projected[:, 2].view(length_array, 1)

    coordinates_projected_normalised = coordinates_projected_normalised[:, :2]
    return coordinates_projected_normalised


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='', help='path of folder containing npy to test')

    params, _ = parser.parse_known_args()

    for folder_name in os.listdir(params.folder_path):
        current_folder_path = os.path.join(params.folder_path, folder_name)
        print(folder_name)
        for file_name in os.listdir(current_folder_path):
            if ".csv" in file_name:
                convert_csv_to_npy(os.path.join(current_folder_path, file_name))

        kpicomputer = ComputeKPI(current_folder_path)
        reprojection_error_results = kpicomputer.compute()
