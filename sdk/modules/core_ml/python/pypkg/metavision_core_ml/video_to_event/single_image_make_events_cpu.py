# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from __future__ import absolute_import

import numpy as np
from numba import prange
from numba import jit

from metavision_sdk_base import EventCD


@jit(nopython=True)
def make_events_cpu(events, ref_values, last_img, last_event_timestamp, log_img, last_img_ts, delta_t, Cps, Cns,
                    refractory_period):
    """
    produce events into AER format

    Args:
        events (np.ndarray): array in format EventCD
        ref_values (np.ndarray): current log intensity state / pixel (H,W)
        last_img (np.ndarray): last image log intensity (H,W)
        last_event_timestamp (int): last image timestamp
        log_img (np.ndarray): current log intensity image (H,W)
        last_img_ts (np.ndarray): last timestamps emitted / pixel (2,H,W)
        delta_t (int): current duration (us) since last image.
        Cps (np.ndarray): array of ON thresholds
        Cns (np.ndarray): array of OFF thresholds
        refractory_period (int): minimum time between 2 events / pixel
    """
    height, width = log_img.shape
    num = 0
    num_iters = 0
    for y in prange(height):
        for x in prange(width):
            itdt = log_img[y, x]
            it = last_img[y, x]

            prev_ref_val = ref_values[y, x]
            pol = 1. if itdt >= it else -1.
            p = 1 if itdt >= it else 0

            # Simulate Thresholds Mismatched
            Cp = Cps[y, x]
            Cn = Cns[y, x]
            C = Cp if pol > 0 else Cn

            if abs(itdt - prev_ref_val) > C:
                current_ref_val = prev_ref_val
                num_events = 0
                all_crossing = False

                while not all_crossing:
                    current_ref_val += pol * C
                    num_events += 1
                    if (pol > 0 and current_ref_val > it and current_ref_val <= itdt) or (
                            pol < 0 and current_ref_val < it and current_ref_val >= itdt):
                        # I can add events
                        edt = (current_ref_val - it) * delta_t / (itdt - it)
                        t = last_img_ts + edt
                        last_stamp_at_xy = last_event_timestamp[y, x]
                        dt = t - last_stamp_at_xy
                        if dt >= refractory_period or last_stamp_at_xy == 0:
                            # we update reference value only if we emit an event
                            events[num]['x'] = x
                            events[num]['y'] = y
                            events[num]['p'] = p
                            events[num]['t'] = t
                            last_event_timestamp[y, x] = t
                            num = min(len(events) - 1, num + 1)

                        ref_values[y, x] = current_ref_val
                    else:
                        all_crossing = True

            num_iters = max(num_iters, num_events)

    return num, num_iters


class EventCPU(object):
    def __init__(self):
        self.max_nb_events = int(1e8)
        self.event_buffer = np.zeros((self.max_nb_events,), dtype=EventCD)
        self.num = 0
        self.num_iters = 0

    def accumulate(
            self, ref_values, last_img, last_event_timestamp, log_img, last_img_ts, delta_t, Cps, Cns,
            refractory_period):
        num, num_iters = make_events_cpu(
            self.event_buffer[self.num:],
            ref_values, last_img, last_event_timestamp, log_img, last_img_ts, delta_t, Cps, Cns, refractory_period)
        self.num += num
        assert self.num < self.max_nb_events - 1, f"Overflow: reached maximum number of events: {self.num}"
        return self.num, num_iters

    def get_max_nb_events(self):
        return self.max_nb_events

    def get_events(self):
        evs = self.event_buffer[:self.num]
        idx = np.argsort(evs['t'])
        evs = evs[idx].copy()
        return evs

    def flush_events(self):
        self.num = 0
        self.num_iters = 0
        self.event_buffer.fill(0)

    def __del__(self):
        del self.event_buffer
