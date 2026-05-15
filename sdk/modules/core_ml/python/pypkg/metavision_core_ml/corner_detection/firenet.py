# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Defines the network's architecture used in
Detecting Stable Keypoints from Events through Image Gradient Prediction
and
Long-Lived Accurate Keypoints in Event Streams
inspired from
Fast Image Reconstruction with an Event Camera
"""
import torch.nn as nn

from metavision_core_ml.core.temporal_modules import ConvRNN, SequenceWise
from metavision_core_ml.core.modules import ConvLayer, PreActBlock


class FireNet(nn.Module):
    def __init__(self, cin=1, cout=1, base=12):
        super().__init__()
        self.conv0 = SequenceWise(PreActBlock(cin, base))
        self.conv1 = SequenceWise(PreActBlock(base, base))
        self.conv3 = ConvRNN(base, base)
        self.conv4 = ConvRNN(base, base)
        self.pred = SequenceWise(
            ConvLayer(base, cout, norm="none", activation="Identity")
        )

    def forward(self, x, mask=None):
        x = self.conv0(x)
        x = self.conv3(x)
        x = self.conv1(x)
        x = self.conv4(x)
        o = self.pred(x)
        return o

    def reset(self, mask):
        self.conv3.reset(mask)
        self.conv4.reset(mask)
