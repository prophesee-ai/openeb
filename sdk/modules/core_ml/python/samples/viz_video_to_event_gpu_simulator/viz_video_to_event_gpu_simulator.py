# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Numba cuda based implementation of event simulator
"""

import argparse
import numpy as np
import torch
import cv2
import time

from torchvision.utils import make_grid
from metavision_core_ml.utils.torch_ops import normalize_tiles
from metavision_core_ml.video_to_event.video_stream_dataset import make_video_dataset
from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_image
from metavision_core_ml.utils.torch_ops import cuda_tick
from profilehooks import profile


def parse_args(only_default_values=False):
    parser = argparse.ArgumentParser(description='Run a batch gpu event based simulator on a video ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', help='path towards image or video dataset')
    parser.add_argument('--threshold-mu', default=0.1, type=float, help="mean contrast threshold")
    parser.add_argument('--threshold-std', default=0.0, type=float, help="std contrast threshold")
    parser.add_argument('--refractory-period', default=0, type=int, help="refractory period")
    parser.add_argument('--batch-size', default=4, type=int, help="grab clips of this duration")
    parser.add_argument('--height', default=240, type=int, help="frame height")
    parser.add_argument('--width', default=320, type=int, help="frame width")
    parser.add_argument('--max-frames-per-video', default=5000, type=int, help="maximum frames to read per video")
    parser.add_argument('--min-frames-per-batch', default=2, type=int, help="minimum frames per batch to construct "
                                                                            "the event tensor")
    parser.add_argument('--max-frames-per-batch', default=10, type=int, help="maximum frames per batch to construct "
                                                                             "the event tensor")
    parser.add_argument('--num-workers', default=2, type=int, help="dataset number of workers")

    parser.add_argument('--device', default='cuda:0', type=str, help="compute device")
    parser.add_argument('--mode', default='event_volume', type=str, help="format returned")
    parser.add_argument("--cutoff-hz", default=0, type=float,
                        help="cutoff frequency for photodiode latency simulation")
    parser.add_argument("--leak-rate-hz", type=float, default=0,
                        help="frequency of reference value leakage")
    parser.add_argument("--shot-noise-hz", default=0, type=float,
                        help="frequency for shot noise events")
    parser.add_argument("--nbins", default=5, type=int,
                        help="voxel_grid nbins")
    parser.add_argument("--split-channels", action='store_true',
                        help="voxel grid split channels")
    parser.add_argument("--delay", default=5, type=int,
                        help="display with a delay in millisecond")
    parser.add_argument("--record-video", action='store_true',
                        help="record a video on the current working directory")
    parser.add_argument('--video-path', default='output.mp4', type=str, help="path of video output (.mp4)")

    return parser.parse_args()


@profile
def test_gpu_simulator(path,
                       threshold_mu, threshold_std,
                       refractory_period,
                       leak_rate_hz, cutoff_hz, shot_noise_hz,
                       num_workers, batch_size,
                       height, width,
                       max_frames_per_video,
                       device, mode, split_channels=False,
                       min_frames_per_batch=2,
                       max_frames_per_batch=10,
                       nbins=10,
                       delay=5,
                       record_video=False,
                       video_path="output.mp4"):
    """
    Fixed Number of Frames/ Video
    """
    print('parameters:', locals())
    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    dl = make_video_dataset(
        path, num_workers, batch_size, height, width, max_frames_per_video - 1, max_frames_per_video,
        min_frames=min_frames_per_batch, max_frames=max_frames_per_batch, rgb=False)
    event_gpu = GPUEventSimulator(batch_size, height, width, threshold_mu,
                                  threshold_std, refractory_period, leak_rate_hz, cutoff_hz, shot_noise_hz)

    event_gpu.to(device)
    dl.to(device)
    pause = False
    last_images = None
    start = time.time()
    if record_video:
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width*2, height))
    for i, batch in enumerate(dl):

        images = batch['images'].squeeze(dim=0)

        first_times = batch['first_times']
        timestamps = batch['timestamps']
        num_frames = batch['video_len']

        if pause and last_images is not None:
            images = last_images

        start = cuda_tick()

        # randomize parameters
        # event_gpu.randomize_broken_pixels(first_times, video_proba=0.1)
        # event_gpu.randomize_thresholds(first_times)
        # event_gpu.randomize_cutoff(first_times)
        # event_gpu.randomize_shot(first_times)
        # event_gpu.randomize_refractory_periods(first_times)

        log_images = event_gpu.dynamic_moving_average(images, num_frames, timestamps, first_times)

        if mode == 'counts':
            idx = 1
            counts = event_gpu.count_events(log_images, num_frames, timestamps, first_times)
        elif mode == 'event_volume':
            voxels = event_gpu.event_volume(log_images, num_frames, timestamps,
                                            first_times, nbins, mode, split_channels=split_channels)
            if split_channels:
                counts = voxels[:, nbins:] - voxels[:, :nbins]
                counts = counts.mean(dim=1)
            else:
                counts = voxels.mean(dim=1)
        else:
            events = event_gpu.get_events(log_images, num_frames, timestamps, first_times)
            counts = event_image(events, batch_size, height, width)

        end = cuda_tick()
        print('total runtime: ', end - start)

        im = 255 * normalize_tiles(counts.unsqueeze(1).float(), num_stds=3)

        im = make_grid(im, nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

        blur_images = torch.exp(log_images) * 255
        first_frames_indices = torch.cat((torch.zeros(1, device=images.device), num_frames.cumsum(0)))[
            :-1].long()
        imin = make_grid(
            blur_images[None, ..., first_frames_indices].permute(3, 0, 1, 2), nrow=nrows).detach().cpu().permute(
            1, 2, 0).numpy().astype(np.uint8)
        final = np.concatenate((im, imin), axis=1)
        cv2.imshow('all', final)
        if record_video:
            out.write(final)
        key = cv2.waitKey(delay)
        if key == 27 or key == ord('q'):
            break
        if key == ord('p'):
            pause = ~pause

        last_images = images
        start = time.time()
    cv2.destroyWindow('all')
    if record_video:
        out.release()


if __name__ == '__main__':
    ARGS = parse_args()
    test_gpu_simulator(**ARGS.__dict__)
