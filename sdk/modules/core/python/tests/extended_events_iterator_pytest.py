# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for ExtendedEventsIterator class
"""
import os
import numpy as np
import pytest

test_dir_path = os.path.dirname(os.path.realpath(__file__))
sample_dir_path = os.path.join(test_dir_path, '..', 'samples', 'metavision_interop')
assert os.path.isdir(sample_dir_path)
import sys
sys.path.append(sample_dir_path)

from extended_events_iterator import ExtendedEventsIterator


def pytestcase_iterator_csv_zip_eth(dataset_dir):
    """Tests initialization of all member variables after creation of RawReader object from a file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "csv_zip_eth", "events.zip")
    mv_iterator = ExtendedEventsIterator(filename, delta_t=10000)

    # WHEN
    height, width = mv_iterator.get_size()
    # THEN
    assert width == 640
    assert height == 480

    list_all_events = []
    for ev in mv_iterator:
        list_all_events.append(ev)

    assert len(list_all_events) == 6
    assert [ev.size for ev in list_all_events] == [2, 4, 1, 0, 0, 1]
    all_ev_np = np.concatenate(list_all_events)

    assert all_ev_np["t"].tolist() == [7000, 8000, 15000, 15002, 16000, 16125, 21000, 52000]
    assert all_ev_np["x"].tolist() == [100, 101, 102, 103, 104, 105, 106, 107]
    assert all_ev_np["y"].tolist() == [279, 278, 277, 276, 275, 274, 273, 272]
    assert all_ev_np["p"].tolist() == [False, True, False, False, True, False, True, False]
