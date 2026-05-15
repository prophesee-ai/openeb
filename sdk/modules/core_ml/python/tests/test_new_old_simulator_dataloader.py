# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
This will only test

(frame+ts) -> events for old & new

but with all noises

"""
import torch
import numpy as np
import cv2
import tqdm
from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator
from metavision_core_ml.video_to_event.simulator import EventSimulator, eps_log
from metavision_core_ml.video_to_event.video_stream_dataset import make_video_dataset
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume
from metavision_core_ml.utils.torch_ops import normalize_tiles
from torchvision.utils import make_grid


class BatchEventSimulator(object):
    """This would be equivalent in old of new code behavior"""

    def __init__(
            self, batch_size, height, width, threshold_mu, threshold_std, refractory_period, leak_rate_hz, cutoff_hz,
            shot_noise_hz):
        self.simulators = []
        for i in range(batch_size):
            simulator = EventSimulator(height, width, threshold_mu, threshold_mu,
                                       refractory_period, threshold_std, cutoff_hz, leak_rate_hz, shot_noise_hz)
            self.simulators.append(simulator)

    def get_events(self, log_images, timestamps, first_times):
        all_events = []
        for i in range(len(log_images)):
            if first_times[i].item():
                self.simulators[i].reset()
            for t in range(log_images.shape[-1]):
                img = log_images[i, ..., t]
                ts = timestamps[i, t].item()
                self.simulators[i].log_image_callback(img.numpy(), ts)
            events = self.simulators[i].get_events()
            self.simulators[i].flush_events()

            events_th = event_cd_to_torch(events)
            events_th[:, 0] = i
            all_events.append(events_th)
        return torch.cat(all_events)


def main(folder,
         num_workers=2,
         batch_size=4,
         height=240,
         width=320,
         max_frames_per_video=150,
         threshold_mu=0.1,
         threshold_std=0,
         refractory_period=10,
         leak_rate_hz=0,
         cutoff_hz=1e6,
         shot_noise_hz=0,
         seed=0,
         min_error_warning=1e-4):

    videos = make_video_dataset(folder, 0, batch_size, height, width,
                                max_frames_per_video//4, max_frames_per_video, False, seed)
    new_simu = GPUEventSimulator(batch_size, height, width, threshold_mu, threshold_std,
                                 refractory_period, leak_rate_hz, cutoff_hz, shot_noise_hz)

    old_simu = BatchEventSimulator(batch_size, height, width, threshold_mu, threshold_std,
                                   refractory_period, leak_rate_hz, cutoff_hz, shot_noise_hz)

    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    for batch in tqdm.tqdm(videos):
        gray = batch['images'].squeeze(1)
        timestamps = batch['timestamps']
        first_times = batch['first_times']
        prev_ts = new_simu.prev_image_ts

        log_gray = new_simu.dynamic_moving_average(gray, timestamps, first_times)
        # log_gray = new_simu.log_images(gray)

        ev1 = old_simu.get_events(log_gray, timestamps, first_times)
        ev2 = new_simu.get_events(log_gray, timestamps, first_times)

        if len(ev1) == len(ev2) and not len(ev1):
            print('no events skipping')
            continue

        # vizu
        start_times = prev_ts*(1-first_times) + timestamps[:, 0] * first_times
        durations = timestamps[:, -1] - start_times

        vol1 = event_volume(ev1, batch_size, height, width, start_times, durations, 5)
        vol2 = event_volume(ev2, batch_size, height, width, start_times, durations, 5)

        diff = (vol1-vol2)
        max_diff = diff.abs().max().item()

        if max_diff < min_error_warning:
            continue

        print('Warning! max diff: ', max_diff)
        im1 = vol1.mean(dim=1)
        im2 = vol2.mean(dim=1)
        im1 = 255*normalize_tiles(im1[:, None])
        im2 = 255*normalize_tiles(im2[:, None])
        im3 = 255*normalize_tiles(torch.exp(log_gray[..., -1])[:, None])

        im1 = make_grid(im1, nrow=nrows).detach().byte().cpu().permute(1, 2, 0).numpy()
        im2 = make_grid(im2, nrow=nrows).detach().byte().cpu().permute(1, 2, 0).numpy()
        im3 = make_grid(im3, nrow=nrows).detach().byte().cpu().permute(1, 2, 0).numpy()

        cat = np.concatenate((im3, im1, im2), axis=1)
        cv2.imshow('old-new', cat)
        cv2.waitKey(5)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
