# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
This script is a preprocessing script to be run on the Atis Corner Dataset preceding
the script compute_homography_reprojection_error. It will create csv files of corner positions
which can later be evaluated.

"""
import os

import argparse
from enum import Enum
import torch
import torch.nn.functional as F
import csv

from metavision_core_ml.corner_detection.lightning_model import CornerDetectionLightningModel
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume
from metavision_core_ml.corner_detection.corner_tracker import CornerTracker, CCLTracker
from metavision_core_ml.corner_detection.utils import update_nn_tracker, save_nn_corners, clean_pred, events_as_pol
from metavision_core.event_io.py_reader import EventDatReader


class Filter(Enum):
    NONE = 0,
    ACTIVITY = 1,
    STC = 2,
    TRAIL = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='', help='Path of folder with dat files')
    parser.add_argument('checkpoint', type=str, default='', help='Checkpoint to evaluate')
    parser.add_argument('--results-path', type=str, default='', help='Path to results_folder')
    parser.add_argument('--run-by-tbin', action='store_true', help='Iterate over time bins to eval the network')
    parser.add_argument('--multiple-delta-t', action='store_true', help='Compare multiple delta_t')
    parser.add_argument('--compare-channel-predictions', action='store_true', help='Compare multiple prediction type')
    parser.add_argument('--cpu', action='store_true', help='Use cpu')

    sigmoid_fn = torch.nn.Sigmoid()

    params, _ = parser.parse_known_args()
    print('params: ', params)

    if params.multiple_delta_t:
        delta_t_list = [5000, 10000, 20000]
    else:
        delta_t_list = [10000]
    if params.compare_channel_predictions:
        channels_to_predict_list = [[9], [0, 5, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    else:
        channels_to_predict_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]  # default to 10 channels prediction
    for delta_t in delta_t_list:
        for channels_to_predict in channels_to_predict_list:
            prediction_string = ""
            for prediction_index in channels_to_predict:
                prediction_string += "_{}".format(prediction_index)
            result_folder = os.path.join(
                params.results_path, "delta_t_{}_prediction".format(delta_t)+prediction_string)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            start_ts = (80000000 // delta_t) * delta_t

            current_time_window = delta_t
            for events_filename in os.listdir(params.path):
                if "_td.dat" not in events_filename:
                    continue
                events_path = os.path.join(params.path, events_filename)
                print("Evaluating file: {}".format(events_path))

                data_reader = EventDatReader(events_path)
                data_reader.seek_time(start_ts)

                height, width = data_reader.get_size()

                device = "cuda" if not params.cpu else "cpu"
                model = CornerDetectionLightningModel.load_from_checkpoint(params.checkpoint)
                model.eval().to(device)
                nbins = model.hparams.event_volume_depth
                in_height, in_width = (height, width)

                time_tolerance = min(delta_t*1.4, 70000/len(channels_to_predict))
                tracker = CornerTracker(time_tolerance=int(time_tolerance))  # time tolerance can be set to 7000
                csv_path = os.path.join(result_folder, events_filename).replace("_td.dat", ".csv")
                print("writing results to: {}".format(csv_path))
                if os.path.exists(csv_path):
                    print("CSV file already exists. Not overwriting")
                    continue
                csv_file = open(csv_path, "w")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["x", "y", "t", "id"])
                ts = start_ts

                while not data_reader.is_done():
                    events = data_reader.load_delta_t(delta_t)
                    ts += delta_t
                    if events is None:
                        continue
                    if len(events) < 2:
                        continue

                    events_th = event_cd_to_torch(events).to(device)
                    start_times = torch.FloatTensor([events['t'][0]]).view(1, ).to(device)
                    durations = torch.FloatTensor([events['t'][-1] - events['t'][0]]).view(1, ).to(device)

                    tensor_th = event_volume(events_th, 1, height, width, start_times, durations, nbins, 'bilinear')

                    tensor_th = F.interpolate(tensor_th, size=(in_height, in_width),
                                              mode='bilinear', align_corners=True)
                    tensor_th = tensor_th.view(1, 1, nbins, in_height, in_width)

                    pred = model.model(tensor_th)
                    pred = sigmoid_fn(pred)

                    pred = clean_pred(pred, threshold=0.3)
                    num_predicted_time_steps = len(channels_to_predict)
                    for index_channel_to_predict in channels_to_predict:
                        interval_size = current_time_window/10
                        num_interval = num_predicted_time_steps - 1 - index_channel_to_predict
                        ts_from_multi_time_steps = ts - num_interval * interval_size

                        y, x = torch.where((pred[0, 0, index_channel_to_predict, :, :] > 0).squeeze())
                        tracker = update_nn_tracker(tracker, x, y, ts_from_multi_time_steps)

                    save_nn_corners(tracker, csv_writer, ts)

                csv_file.close()


if __name__ == '__main__':
    with torch.no_grad():
        main()
