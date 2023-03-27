# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for conversion EventCD np.arrays to torch tensors (N,5) format
"""

from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, events_cd_list_to_torch, tensor_to_cd_events
from metavision_sdk_base import EventCD

import numpy as np


def pytestcase_event_cd_to_torch_tensor_and_back():
    N = 1000
    events = np.empty(N, dtype=EventCD)
    events["x"] = np.random.randint(0, 640, size=N)
    events["y"] = np.random.randint(0, 480, size=N)
    events["p"] = np.random.randint(0, 2, size=N)
    events["t"] = np.random.randint(0, 20000, size=N)
    events = events[np.argsort(events["t"])]

    events_torch = event_cd_to_torch(events)
    assert events_torch.shape == (N, 5)
    assert (events_torch[:, 0] == 0).all()
    assert (events_torch[:, 1].numpy() == events["x"]).all()
    assert (events_torch[:, 2].numpy() == events["y"]).all()
    assert (events_torch[:, 3].numpy() == events["p"] * 2 - 1).all()
    assert (events_torch[:, 4].numpy() == events["t"]).all()

    events_np = tensor_to_cd_events(events_torch, batch_size=1)
    assert len(events_np) == 1
    assert (events_np[0] == events).all()


def pytestcase_batch_event_cd_to_torch_tensor_and_back():
    events_list = []
    B = 10
    for i in range(B):
        N = np.random.randint(1000, 2000)
        events = np.empty(N, dtype=EventCD)
        events["x"] = np.random.randint(0, 640, size=N)
        events["y"] = np.random.randint(0, 480, size=N)
        events["p"] = np.random.randint(0, 2, size=N)
        events["t"] = np.random.randint(0, 20000, size=N)
        events = events[np.argsort(events["t"])]
        events_list.append(events)

    events_torch = events_cd_list_to_torch(events_list)
    nb_total_events = sum([events.size for events in events_list])
    assert events_torch.shape == (nb_total_events, 5)
    for i in range(B):
        assert (events_torch[:, 0] == i).sum() == events_list[i].size
    events_list_from_torch = tensor_to_cd_events(events_torch, batch_size=B)
    assert len(events_list_from_torch) == B
    for i in range(B):
        assert (events_list_from_torch[i] == events_list[i]).all()
