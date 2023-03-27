# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for EventFrameIterator class
"""
from metavision_core.event_io import EventFrameIterator

import os
import numpy as np


def pytestcase_diff(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "diff3d.raw")
    assert os.path.isfile(filename)
    mv_it = EventFrameIterator(filename)
    assert mv_it.get_frame_type() == "DIFF3D"
    assert mv_it.get_size() == (320, 320)
    for frame_idx, frame in enumerate(mv_it):
        assert frame.dtype == np.int8
        assert frame.shape == (320, 320)
    assert frame_idx == 300


def pytestcase_histo(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "histo3d.raw")
    assert os.path.isfile(filename)
    mv_it = EventFrameIterator(filename)
    assert mv_it.get_frame_type() == "HISTO3D"
    assert mv_it.get_size() == (320, 320)
    for frame_idx, frame in enumerate(mv_it):
        assert frame.dtype == np.uint8
        assert frame.shape == (320, 320, 2)
    assert frame_idx == 301


def pytestcase_histo_padding(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "histo3d_padding.raw")
    assert os.path.isfile(filename)
    mv_it = EventFrameIterator(filename)
    assert mv_it.get_frame_type() == "HISTO3D"
    assert mv_it.get_size() == (320, 320)
    for frame_idx, frame in enumerate(mv_it):
        assert frame.dtype == np.uint8
        assert frame.shape == (320, 320, 2)
    assert frame_idx == 301
