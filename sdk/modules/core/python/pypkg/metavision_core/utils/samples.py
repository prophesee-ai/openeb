# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Utility functions relative to samples
"""
import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import sys
import time

# If URL of a dataset is updated, check if it also listed in doc/source/datasets.rst
# and if so, update the URL in both places.


SAMPLES_DICT = {

    # automotive/
    # -----------
    "driving_sample.raw": "https://dataset.prophesee.ai/index.php/s/nVcLLdWAnNzrmII/download",
    "pedestrians.raw": "https://dataset.prophesee.ai/index.php/s/fB7xvMpE136yakl/download",
    # blinking/
    "blinking_leds_td.dat": "https://dataset.prophesee.ai/index.php/s/Gk6SbZNX3fHZnLw/download",
    "calib_propheshield_parking.10_sec.raw": "https://dataset.prophesee.ai/index.php/s/Rp7ngsn7iDlNAxP/download",

    # counting/
    # ---------
    "80_balls.raw": "https://dataset.prophesee.ai/index.php/s/2j8xCyufDLhi7sd/download",
    "195_falling_particles.raw": "https://dataset.prophesee.ai/index.php/s/kSfKSZfCLKPIblT/download",
    # generic/
    # --------
    "laser.raw": "https://dataset.prophesee.ai/index.php/s/bX0CnMiQ2XVGGjv/download",
    "spinner.dat": "https://dataset.prophesee.ai/index.php/s/YAri3vpPZHhEZfc/download",
    "spinner.raw": "https://dataset.prophesee.ai/index.php/s/mwiILym2zD8ud2b/download",
    "hand_spinner.raw": "https://dataset.prophesee.ai/index.php/s/nOcN4Fzlv5qCAxt/download",
    "traffic_monitoring.raw": "https://dataset.prophesee.ai/index.php/s/GxgIdzXvdU0f1Xo/download",

    # jet_monitoring/
    # ---------------
    "200_jets.raw": "https://dataset.prophesee.ai/index.php/s/LCheVtNeIoJptHF/download",

    # spatter_tracking/
    # ------------------
    "sparklers.raw": "https://dataset.prophesee.ai/index.php/s/95dTpv4ZJ60jyI5/download",

    # vibration/
    # ----------
    #
    "monitoring_40_50hz.raw": "https://dataset.prophesee.ai/index.php/s/s5DFqzVQhlaU8Y5/download",

    # machine_learning/
    # ----------
    "day_200618_170002_0_20.raw": "https://dataset.prophesee.ai/index.php/s/tI3smokMz0xVX6R/download",
    "day_200618_mask.png": "https://dataset.prophesee.ai/index.php/s/j9VHorWyLOkbVnG/download",
    "MNIST.zip": "https://dataset.prophesee.ai/index.php/s/jKHZ4zRHIllSLE1/download",
    "mini-dataset": "https://dataset.prophesee.ai/index.php/s/ScqMu02G5pdYKPh/download"
}


def get_sample(sample_name, folder='.'):
    """
    Ensures that a file of name `sample_name` is indeed in `folder`.

    If not, attempts to download it from Prophesee's public sample server.

    Args:
        sample_name (string): Basename of the requested sample.
        folder (string): Path to a folder where data should be saved.
    """
    assert os.path.isdir(folder), f"{folder} is not a directory !"
    assert sample_name in SAMPLES_DICT, f"No sample {sample_name}, must be one of " + ", ".join(SAMPLES_DICT)

    sample_path = os.path.join(folder, sample_name)
    if not os.path.exists(sample_path):
        print("Downloading file {} -> {}".format(SAMPLES_DICT[sample_name], sample_path))

        def reporthook(count, block_size, total_size):
            if count % 10 != 0:
                return
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            if duration == 0:
                return
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urlretrieve(SAMPLES_DICT[sample_name], filename=sample_path, reporthook=reporthook)


def get_all_samples(folder="."):
    """Ensures that all public samples are downloaded in the given folder

    Args:
        folder (string): Path to a folder where data should be saved.
    """
    for sample_name in SAMPLES_DICT:
        get_sample(sample_name, folder=folder)
