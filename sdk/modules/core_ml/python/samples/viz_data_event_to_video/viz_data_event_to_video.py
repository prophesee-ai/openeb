# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Visualize the event-to-video dataloader
"""

import sys
import time
import numpy as np
import cv2
from metavision_core_ml.event_to_video.gpu_esim import GPUEBSIM
from metavision_core_ml.utils.torch_ops import normalize_tiles
from torchvision.utils import make_grid


def show_dataset(dataloader, batch_size, delay=5, verbose=False):
    start = 0
    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    for batch_num, batch in enumerate(dataloader):
        if verbose:
            sys.stdout.write(f"\rbatch runtime: {time.time() - start:4.4f}")
            sys.stdout.flush()

        x = batch["inputs"].detach().cpu()
        y = batch["images"].detach().cpu()

        for t in range(len(x)):
            gy = make_grid(y[t], nrow=nrows).permute(1, 2, 0).numpy().astype(np.uint8)

            # mean
            xt = x[t].mean(dim=1, keepdims=True)
            xt = 255 * normalize_tiles(xt, num_stds=6, real_min_max=True)
            gx = make_grid(xt, nrow=nrows).permute(1, 2, 0).numpy().astype(np.uint8)

            cat = np.concatenate((gx, gy), axis=1).copy()

            cv2.putText(
                cat, "frame#" + str(batch_num * len(x) + t),
                (10, cat.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow('data', cat)
            cv2.waitKey(delay)

        start = time.time()


def main(folder, num_workers=2, batch_size=4, batch_times=4, event_volume_depth=5, height=240, width=320,
         min_frames_per_video=50, max_frames_per_video=5000, batch_sampling_mode='random', batch_min_frames=1,
         batch_max_frames=10, batch_min_delta_t=1000, batch_max_delta_t=5000, device='cuda:0', verbose=False,
         randomize_noises=False):
    """
    Visualises a DataLoader for event to video training.

    Args:
        folder (string): folder containing the mp4 videos (and optional _ts.npy timestamps files)/
        num_workers (int): number of processes used.
        batch_size (int): number of videos loaded simultaenously
        batch_times (int): number of time bins.
        event_volume_depth (int): number of channels in the event cube.
        height (int): height of each video in pixels.
        width (int): width of each video in pixels.
        min_frames_per_video (int): min number of frames for each video.
        max_frames_per_video (int): max number of frames for each video.
        batch_sampling_mode (bool): random, frames or delta_t
        batch_min_frames (int): min frames per batch
        batch_max_frames (int): max frames per batch
        batch_min_delta_t (int): min duration per batch
        batch_max_delta_t (int): max duration per batch
        device (string): either cuda or cpu (must be a valid descriptor for a torch.device)
        verbose (boolean): wether to print batch loading times
        randomize_noises (boolean): add noises
    """
    dataloader = GPUEBSIM.from_params(
        folder,
        num_workers,
        batch_size,
        batch_times,
        event_volume_depth,
        height,
        width,
        min_frames_per_video,
        max_frames_per_video,
        batch_sampling_mode,
        batch_min_frames,
        batch_max_frames,
        batch_min_delta_t,
        batch_max_delta_t,
        randomize_noises,
        device)
    show_dataset(dataloader, batch_size, 5, verbose=verbose)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
