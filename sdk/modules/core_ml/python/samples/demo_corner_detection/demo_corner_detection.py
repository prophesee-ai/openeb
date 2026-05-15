# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Corner Detection Demo Script
"""


import numpy as np
import argparse
import torch
import csv

from metavision_core_ml.corner_detection.lightning_model import CornerDetectionLightningModel
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume
from metavision_core_ml.utils.show_or_write import ShowWrite
from metavision_core_ml.corner_detection.corner_tracker import CornerTracker
from metavision_core_ml.corner_detection.utils import clean_pred, update_nn_tracker, save_nn_corners
from metavision_core.event_io import EventsIterator


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='', help='path of events')
    parser.add_argument('checkpoint', type=str, default='', help='checkpoint to evaluate')
    parser.add_argument('--video-path', type=str, default='', help='path to video')
    parser.add_argument('--start-ts', type=int, default=0, help='start timestamp')
    parser.add_argument('--mode', type=str, default='delta_t',
                        choices=['n_events', 'delta_t', 'mixed', 'adaptive'], help='how to cut events')
    parser.add_argument('--n_events', type=int, default=2000, help='num of events to load')
    parser.add_argument('--delta-t', type=int, default=5000, help='time in us to load')
    parser.add_argument('--max-duration', type=int, default=-1, help='run for this duration')
    parser.add_argument('--thr-var', type=float, default=3e-5, help='threshold variance for adaptive rate')
    parser.add_argument('--cpu', action='store_true', help='if true use cpu and not cuda')
    parser.add_argument('--save-corners', action='store_true', help='also write the corners to a CSV file')
    parser.add_argument('--use-multi-time-steps', action='store_true', help='Do not re-aggregate predictions')
    parser.add_argument('--load-by-n-events', action='store_true', help='Load n events instead of delta_t')
    parser.add_argument('--show', action='store_true', help='Show results')

    sigmoidfn = torch.nn.Sigmoid()

    params, _ = parser.parse_known_args(raw_args)
    print('params: ', params)

    events_iterator = EventsIterator(params.path, start_ts=params.start_ts,
                                     mode=params.mode, delta_t=params.delta_t, n_events=params.n_events)

    window_name = None
    if params.show:
        window_name = "Corner Detection"
    show_write = ShowWrite(window_name, params.video_path)
    print("writing video to : {}".format(params.video_path))

    height, width = events_iterator.get_size()
    print('original size: ', height, width)

    device = 'cpu' if params.cpu else 'cuda'
    model = CornerDetectionLightningModel.load_from_checkpoint(params.checkpoint)
    model.eval().to(device)
    nbins = model.hparams.event_volume_depth
    print('Nbins: ', nbins)

    tracker = CornerTracker(time_tolerance=7000, distance_tolerance=3)
    if params.save_corners:
        video_ext = params.video_path[-4:]
        assert video_ext in [".avi", ".mp4"], video_ext
        csv_file = open(params.video_path.replace(video_ext, ".csv"), "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y", "t", "id"])
    ts = params.start_ts

    for index_events, events in enumerate(events_iterator):
        if events is None or len(events) < 2:
            continue

        if params.mode == "delta_t":
            current_time_window = params.delta_t
        else:
            current_time_window = events["t"][-1] - events["t"][0]
        ts += current_time_window
        print(f"Running corner detection: ts {ts}")
        events_th = event_cd_to_torch(events).to(device)
        start_times = torch.FloatTensor([events['t'][0]]).view(1, ).to(device)
        durations = torch.FloatTensor([events['t'][-1] - events['t'][0]]).view(1, ).to(device)

        tensor_th = event_volume(events_th, 1, height, width, start_times, durations, nbins, 'bilinear')

        tensor_th = tensor_th.view(1, 1, nbins, height, width)

        pred = model.model(tensor_th)
        if not params.use_multi_time_steps:
            pred = pred.sum(2)
            pred = pred.unsqueeze(2)

        pred = sigmoidfn(pred)
        image = model.image_from_events(tensor_th)
        if params.use_multi_time_steps:
            heat_map_image = model.make_heat_map_image(pred, divide_max=True)
        else:
            heat_map_image = model.make_heat_map_image(pred, divide_max=False)

        pred = clean_pred(pred, threshold=0.1)

        if params.use_multi_time_steps:
            num_predicted_time_steps = pred.shape[2]
            for index_time_step in range(pred.shape[2]):
                y, x = torch.where((pred[0, 0, index_time_step, :, :] > 0).squeeze())

                ts_from_multi_time_steps = ts - ((num_predicted_time_steps - index_time_step)
                                                 * current_time_window / num_predicted_time_steps)
                tracker = update_nn_tracker(tracker, x, y, ts_from_multi_time_steps)
                if params.save_corners:
                    save_nn_corners(tracker, csv_writer, ts)
        else:
            y, x = torch.where((pred[0, 0, 0, :, :] > 0).squeeze())

            tracker = update_nn_tracker(tracker, x, y, ts)
            if params.save_corners:
                save_nn_corners(tracker, csv_writer, ts)

        if index_events % (20000 // params.delta_t) == 0:
            image = torch.cat([image] * 3, axis=2).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            image = tracker.show(image)
            key = show_write(np.concatenate([image, heat_map_image], axis=1))
            if key == 27:
                break
            if key == ord('p'):
                pause = not pause

    if params.save_corners:
        csv_file.close()


if __name__ == '__main__':
    with torch.no_grad():
        main()
