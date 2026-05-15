# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
RosBag Reader

This is an example on how to read a different format.
Here we use the "rospy" library to read .bag files:
https://github.com/rospypi/simple

You can install the library like so:
python3 -m pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
"""

import rosbag
import numpy as np

from itertools import islice
from metavision_sdk_base import EventCD


class RosBaseReader(object):
    """
    BaseReader for .bag format.
    We write a base iterator returning numpy buffers of type EventCD.
    """

    def __init__(self, rosbag_path, event_topic='/dvs/events'):
        self.bag = rosbag.Bag(rosbag_path, 'r')
        self.event_topic = event_topic
        topics = self.bag.get_type_and_topic_info().topics
        assert event_topic in topics, 'event topic is not present'
        self.t0 = None
        self.height, self.width = -1, -1
        for topic, msg, t in islice(self.bag.read_messages(), 6):
            if hasattr(msg, "width"):
                self.height = msg.height
                self.width = msg.width
        if self.height < 0 or self.width < 0:
            raise BaseException("No Message with height or width fields")

    def is_done(self):
        return False

    def __del__(self):
        pass

    def get_size(self):
        return self.height, self.width

    def seek_time(self, ts):
        if ts != 0:
            raise Exception('time seek not implemented in rospy')

    def __iter__(self):
        t0 = None
        for topic, msg, t in self.bag.read_messages():
            if topic == self.event_topic:
                evs = msg.events
                num = len(evs)
                if t0 is None:
                    t0 = evs[0].ts.to_nsec() / 1000
                event_buffer = np.zeros((num,), dtype=EventCD)
                for n, ev in enumerate(evs):
                    event_buffer[n]['x'] = ev.x
                    event_buffer[n]['y'] = ev.y
                    event_buffer[n]['p'] = ev.polarity
                    event_buffer[n]['t'] = ev.ts.to_nsec() / 1000 - t0
                yield event_buffer
