# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# pylint: disable=E1101

"""
Unit tests for Raw files headers
"""
import os

from metavision_core.event_io.raw_info import raw_file_header, is_event_raw, is_event_frame_raw, raw_histo_header_bits_per_channel


def pytestcase_check_header_events_raw(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "blinking_gen4_with_ext_triggers.raw")
    assert os.path.isfile(filename)
    assert is_event_raw(filename)
    assert not is_event_frame_raw(filename)


def pytestcase_check_header_event_frames_diff(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "diff3d.raw")
    assert os.path.isfile(filename)
    assert not is_event_raw(filename)
    assert is_event_frame_raw(filename)
    header_dic = raw_file_header(filename)
    assert "format" in header_dic
    assert header_dic["format"] == "DIFF3D"


def pytestcase_check_header_event_frame_histo(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "histo3d.raw")
    assert os.path.isfile(filename)
    assert not is_event_raw(filename)
    assert is_event_frame_raw(filename)
    header_dic = raw_file_header(filename)
    assert "format" in header_dic
    assert header_dic["format"] == "HISTO3D"
    bits_neg, bits_pos = raw_histo_header_bits_per_channel(filename)
    assert (bits_neg, bits_pos) == (4, 4)
