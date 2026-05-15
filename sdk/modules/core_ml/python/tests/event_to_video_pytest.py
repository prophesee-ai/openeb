# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import torch
from metavision_core_ml.event_to_video.event_to_video import EventToVideo


def pytestcase_event_to_video():
    net = EventToVideo(10, 1, num_layers=3, base=16, cell='lstm')

    t, b, c, h, w = 4, 3, 10, 64, 64
    x = torch.randn(t, b, c, h, w)
    y = net(x)
    z = net.predict_gray(y)
    assert z.shape == torch.Size((t, b, 1, h, w))
