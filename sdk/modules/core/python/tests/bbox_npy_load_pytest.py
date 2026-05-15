# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for EventBboxNpyReader class loading bbox npy
"""
import os
from metavision_core.event_io.box_npy_reader import EventBboxNpyReader
from metavision_sdk_core import EventBbox


def pytestcase_load_old_box_npy_file(dataset_dir):
    path = os.path.join(dataset_dir, 'openeb', 'core', 'event_io', 'old_bbox.npy')
    box_reader = EventBboxNpyReader(path)
    boxes = box_reader.load_delta_t(10000000)
    assert boxes.dtype == EventBbox
