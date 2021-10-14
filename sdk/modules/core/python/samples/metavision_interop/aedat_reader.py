# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
AEDAT Reader

This is an example on how to read a different format.
Here we use the "aedat" library:
https://github.com/neuromorphicsystems/aedat

You can install the library like so:
python3 -m pip install aedat
"""

import aedat
import numpy as np
from metavision_sdk_base import EventCD


class AEDATBaseReader(object):
    """
    BaseReader for .aedat4 format.
    We write a base iterator returning numpy buffers of type EventCD.
    """

    def __init__(self, aedat_path):
        self.decoder = aedat.Decoder(aedat_path)
        self.t0 = None

    def is_done(self):
        return False

    def __del__(self):
        pass

    def get_size(self):
        for packet in self.decoder:
            if 'frame' in packet:
                return packet['frame']['height'], packet['frame']['width']
        raise Exception('frame size not found')

    def seek_time(self, ts):
        if ts != 0:
            raise Exception('time seek in aedat not implemented')

    def __iter__(self):
        t0 = None
        for packet in self.decoder:
            if 'events' in packet:
                events = packet['events']
                num = len(events)
                if t0 is None:
                    t0 = events['t'][0]
                event_buffer = np.zeros((num,), dtype=EventCD)
                event_buffer['t'][:num] = (events['t'] - t0)
                event_buffer['x'][:num] = events['x']
                event_buffer['y'][:num] = events['y']
                event_buffer['p'][:num] = events['on']
                yield event_buffer
            else:
                continue
