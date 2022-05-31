# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Function to visualize events
"""

import numpy as np
from numba import jit

BG_COLOR = np.array([30, 37, 52], dtype=np.uint8)
POS_COLOR = np.array([216, 223, 236], dtype=np.uint8)
NEG_COLOR = np.array([64, 126, 201], dtype=np.uint8)


def viz_events(events, height, width, img=None):
    """Creates a RGB frame representing the events given as input.
    Args:
        events (np.ndarray): structured array containing events
        height (int): Height of the sensor in pixels
        width (int): width of the sensor in pixels
        img (np.ndarray): optional image of size (height, width, 3) and dtype unint8 to avoid reallocation
    Returns:
        output_array (np.ndarray): Array of shape (height, width, 3)
    """
    if img is None:
        img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    else:
        img[...] = BG_COLOR
    _viz_events(events, img)
    return img


@jit
def _viz_events(events, img):
    for i in range(events.shape[0]):
        x = int(events[i]["x"])
        y = int(events[i]["y"])
        if events[i]["p"] > 0:
            img[y, x, :] = POS_COLOR
        else:
            img[y, x, :] = NEG_COLOR
