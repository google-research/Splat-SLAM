# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False

        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, kernel_size=(3, 3), padding=(1, 1))
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, kernel_size=(3, 3), padding=(1, 1))
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, kernel_size=(3, 3), padding=(1, 1))

        self.w = nn.Conv2d(h_planes, h_planes, kernel_size=(1, 1), padding=(0, 0))

        self.convz_glo = nn.Conv2d(h_planes, h_planes, kernel_size=(1, 1), padding=(0, 0))
        self.convr_glo = nn.Conv2d(h_planes, h_planes, kernel_size=(1, 1), padding=(0, 0))
        self.convq_glo = nn.Conv2d(h_planes, h_planes, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        b, c, h, w = net.shape
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h*w).mean(dim=-1, keepdim=True).view(b, c, 1, 1)

        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + self.convq_glo(glo))

        net = (1 - z) * net + z * q

        return net