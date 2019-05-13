# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class inception(nn.Module):
    def __init__(self, input_size, config):
        self.config = config
        super(inception, self).__init__()
        self.convs = nn.ModuleList()

        # Base 1*1 conv layer
        self.convs.append(nn.Sequential(
            nn.Conv2d(input_size, config[0][0], 1),
            nn.BatchNorm2d(config[0][0], affine=False),
            nn.ReLU(True),
        ))

        # Additional layers
        for i in range(1, len(config)):
            filt = config[i][0]
            pad = int((filt-1)/2)
            out_a = config[i][1]
            out_b = config[i][2]
            conv = nn.Sequential(
                nn.Conv2d(input_size, out_a, 1),
                nn.BatchNorm2d(out_a, affine=False),
                nn.ReLU(True),
                nn.Conv2d(out_a, out_b, filt, padding=pad),
                nn.BatchNorm2d(out_b, affine=False),
                nn.ReLU(True)
            )
            self.convs.append(conv)

    def __repr__(self):
        return "inception"+str(self.config)

    def forward(self, x):
        ret = []
        for conv in (self.convs):
            ret.append(conv(x))
        return torch.cat(ret, dim=1)


class Channels1(nn.Module):
    def __init__(self):
        super(Channels1, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
            )
        )  # EE
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # EEE

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class Channels2(nn.Module):
    def __init__(self):
        super(Channels2, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])
            )
        )  # EF
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                Channels1(),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # EE1EF

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class Channels3(nn.Module):
    def __init__(self):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                Channels2(),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # BD2EG
        self.list.append(
            nn.Sequential(
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])
            )
        )  # BC

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class Channels4(nn.Module):
    def __init__(self):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                Channels3(),
                inception(128, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]]),
                inception(128, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # BB3BA
        self.list.append(
            nn.Sequential(
                inception(128, [[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])
            )
        )  # A

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class HourglassModel(nn.Module):
    def __init__(self, num_input):
        super(HourglassModel, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(num_input, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            Channels4(),
        )

        uncertainty_layer = [
            nn.Conv2d(64, 1, 3, padding=1), torch.nn.Sigmoid()]
        self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
        self.pred_layer = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, input_):
        pred_feature = self.seq(input_)

        pred_d = self.pred_layer(pred_feature)
        pred_confidence = self.uncertainty_layer(pred_feature)

        return pred_d, pred_confidence
