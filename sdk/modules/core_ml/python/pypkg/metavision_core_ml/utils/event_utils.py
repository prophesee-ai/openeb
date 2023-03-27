# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


def split_events(events, split_times):
    """
    Splits events by a split_times

    Args:
        events (tensor): N,5 (b,x,y,p,t)
        split_times: (B,) split times per sequence
    """
    bs = events[:, 0].long()
    split_times = split_times[bs]
    mask = events[:, 4] >= split_times
    left_events = events[~mask]
    right_events = events[mask]
    return left_events, right_events
