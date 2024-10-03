# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for the sample "demo_event_to_video.py"
"""
import os
import sys
import numpy as np
import pytest
import torch

test_dir_path = os.path.dirname(os.path.realpath(__file__))
sample_dir_path = os.path.join(test_dir_path, '..', 'samples', 'demo_event_to_video')
assert os.path.isdir(sample_dir_path)
sys.path.append(sample_dir_path)


from demo_event_to_video import parse_args, run


class TestEvent2VideoInferenceMain(object):

    def pytestcase_CLI_test(self, tmpdir, dataset_dir):
        """checks that the box are split correctly."""
        # GIVEN
        video_path = os.path.join(dataset_dir, 'openeb', 'gen4_evt2_hand.raw')
        model_path = os.path.join(dataset_dir, '..', 'sdk', 'modules', 'core_ml', 'models',
                                  'e2v.ckpt')

        # WHEN
        params = parse_args([video_path, model_path, '--height_width', '240',  '320', '--mode', 'delta_t',
                             '--delta_t', '50000', '--max_duration', '2000000', '--no_window', '--cpu',
                             '--video_path', os.path.join(tmpdir, 'e2vid.mp4')])
        run(params)
        # THEN
        # just validate we did not crash
