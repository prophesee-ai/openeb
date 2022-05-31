# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for EventsIterator class
"""
import os
import math
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import H5EventsWriter


def pytestcase_equivalence(tmpdir, dataset_dir):
    """Tests initialization of all member variables after creation of RawReader object from a file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")

    raw_iterator = EventsIterator(filename, start_ts=0, delta_t=10000,
                                  max_duration=1e7, relative_timestamps=False)
    # WHEN (we write h5 file)
    height, width = raw_iterator.get_size()

    out_path = str(tmpdir.join(os.path.basename(os.path.splitext(filename)[0] + '.h5')))
    raw_event_buffers = []
    with H5EventsWriter(out_path, height, width, "zlib") as h5_writer:
        for i, events in enumerate(raw_iterator):
            h5_writer.write(events.copy())
            if len(events):
                raw_event_buffers.append(events.copy())

    # THEN
    h5_iterator = EventsIterator(out_path, start_ts=0, delta_t=10000,
                                 max_duration=1e7, relative_timestamps=False)
    h5_event_buffers = []
    for h5_events in h5_iterator:
        h5_event_buffers.append(h5_events.copy())

    assert len(raw_event_buffers) == len(h5_event_buffers)

    for raw_events, h5_events in zip(raw_event_buffers, h5_event_buffers):
        for k in ['x', 'y', 'p', 't']:
            np.testing.assert_equal(raw_events[k], h5_events[k])
