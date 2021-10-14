# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for AdaptiveRateEventsIterator class
"""
import os
import math
import numpy as np

from metavision_core.event_io import AdaptiveRateEventsIterator


def pytestcase_iterator_loop(tmpdir, dataset_dir):
    """Tests initialization and looping"""
    # GIVEN
    filename = os.path.join(dataset_dir, "openeb", "gen31_timer.raw")
    mv_iterator = AdaptiveRateEventsIterator(filename)

    # WHEN
    height, width = mv_iterator.get_size()
    # THEN
    assert width == 640
    assert height == 480

    for i, events in enumerate(mv_iterator):
        assert events.size > 20000
        if i >= 10:
            break
