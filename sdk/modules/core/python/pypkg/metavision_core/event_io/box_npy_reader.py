# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np

from . import EventNpyReader
from metavision_sdk_core import EventBbox


class EventBboxNpyReader(EventNpyReader):
    """
    EventBboxNpyReader class to read NPY long files.

    Attributes:
        path (string): Path to the file being read
        current_time (int): Indicating the position of the cursor in the file in us
        duration_s (int): Indicating the total duration of the file in seconds

    Args:
        event_file (str): file containing events
    """

    def __init__(self, event_file):
        super().__init__(event_file)

    def open_file(self):
        super().open_file()
        self._decode_dtype = EventBbox
