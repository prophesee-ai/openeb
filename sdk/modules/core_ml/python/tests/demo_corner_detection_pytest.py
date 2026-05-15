# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Test the demo script for corner detection
"""
import os
import sys
from metavision_utils.shell_tools import execute_cmd


def pytestcase_demo_corner_detection(tmpdir, dataset_dir):
    """
    Testing if the demo of the corner detection works
    """
    video_path = os.path.join(tmpdir, "demo_video.avi")
    datfile = os.path.join(dataset_dir, "openeb/core_ml/corner_detection/guernica_small_for_pytest.dat")
    
    models_dir = os.path.join(dataset_dir, '..', 'sdk', 'modules', 'core_ml', 'models')
    checkpoint = os.path.join(models_dir, "corner_detection_10_heatmaps.ckpt")

    assert os.path.exists(checkpoint)
    assert os.path.exists(datfile)

    samples_dir = os.path.join(dataset_dir, '..', 'sdk', 'modules', 'core_ml', 'python', 'samples')
    demo_path = os.path.join(samples_dir, "demo_corner_detection/demo_corner_detection.py")
    assert os.path.exists(demo_path)
    execute_cmd('{} {} \
                         {} \
                         {} \
                         --video-path {} \
                         --use-multi-time-steps \
                         --save-corners \
                         --cpu'.format(sys.executable, demo_path, datfile, checkpoint, video_path),
                env=os.environ.copy())

    assert os.path.exists(video_path)
    assert os.path.exists(video_path.replace(".avi", ".csv"))
