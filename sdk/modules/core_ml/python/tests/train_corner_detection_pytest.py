# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Test the training script for corner detection
"""
import os
import sys
from metavision_utils.shell_tools import execute_cmd


def pytestcase_training_corner_detection(tmpdir, dataset_dir):
    """
    Testing if the training of the corner detection works
    """
    image_dir = os.path.join(dataset_dir, "openeb/core_ml/corner_detection/images")
    log_dir = os.path.join(tmpdir, "corner_detection_pytest")

    samples_dir = os.path.join(dataset_dir, '..', 'sdk', 'modules', 'core_ml', 'python', 'samples')
    train_path = os.path.join(samples_dir, "train_corner_detection/train_corner_detection.py")
    assert os.path.exists(train_path)

    execute_cmd('{} {} \
                --root_dir {} \
                --dataset_path {} \
                --number_of_heatmaps 10 \
                --height 36 \
                --width 48 \
                --batch_size 2 \
                --precision 32 \
                --limit_train_batches 2 \
                --randomize_noises \
                --cout 10 \
                --epochs 2 \
                --demo_iter 5 \
                --demo_every 1 \
                --num_workers 0 \
                --cpu'.format(sys.executable, train_path, log_dir, image_dir),
                env=os.environ.copy())

    checkpoints_path = os.path.join(log_dir, "checkpoints")
    assert os.path.exists(checkpoints_path)
    assert len(os.listdir(checkpoints_path)) > 0

    assert os.path.exists(os.path.join(log_dir, "videos/video_train_1.mp4"))
    assert os.path.exists(os.path.join(log_dir, "videos/video_val_1.mp4"))
