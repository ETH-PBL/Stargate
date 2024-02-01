#
# Copyright (C) 2022-2024 ETH Zurich
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0
# ======================================================================
#
# Authors: 
# Konstantin Kalenberg, ETH Zurich (kkalenbe@ethz.ch)
# Hanna Müller ETH Zurich (hanmuell@iis.ee.ethz.ch)
# Tommaso Polonelli, ETH Zurich (tommaso.polonelli@pbl.ee.ethz.ch)
# Alberto Schiaffino, ETH Zurich (aschiaffino@ethz.ch)
# Vlad Niculescu, ETH Zurich (vladn@ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@ethz.ch)
# Michele Magno, ETH Zurich (michele.magno@pbl.ee.ethz.ch)
# Luca Benini, ETH Zurich (lbenini@iis.ee.ethz.ch)
#
# Classes for deep learning model used in my master thesis
# Adapted from https://github.com/pulp-platform/pulp-dronet/blob/master/pulp-dronet-v2/model/dronet_v2_gapflow.py


import torch
import torch.nn as nn


"""
Double convolutional block used in GateClassifier class
"""
class DoubleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1,
                               bias=True, padding_mode='zeros')

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                               groups=1, bias=True, padding_mode='zeros')

        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu1 = nn.ReLU6(inplace=False)

        self.relu2 = nn.ReLU6(inplace=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x



"""
GateClassifier network using camera + tof as input and outputting probability of seeing a gate [0, 1] as a classification task
"""
class GateClassifier(nn.Module):
    def __init__(self, num_channels_start=4, dropout_p=0.0):
        super(GateClassifier, self).__init__()

        # conv 5x5, 1, num_channels_start, 168x168,
        self.conv_camera_layer_1 = nn.Conv2d(in_channels=1, out_channels=num_channels_start, kernel_size=5, stride=2, padding=2, dilation=1,
                                             groups=1, bias=True, padding_mode='zeros')

        self.bn_camera_1 = nn.BatchNorm2d(num_features=num_channels_start, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu_camera_1 = nn.ReLU6(inplace=False)

        # max pooling 2x2, num_channels_start, num_channels_start, 42x42
        self.camera_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # block, num_channels_start, num_channels_start, 21x21
        self.camera_block_1 = DoubleConvBlock(num_channels_start, num_channels_start)

        # conv 3x3, 1, num_channels_start, 21x21, /1
        self.tof_layer_1 = nn.Conv2d(in_channels=1, out_channels=num_channels_start, kernel_size=3, stride=1, padding=1, dilation=1,
                                     groups=1, bias=True, padding_mode='zeros')

        self.bn_tof_1 = nn.BatchNorm2d(num_features=num_channels_start, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu_tof_1 = nn.ReLU6(inplace=False)

        self.combined_block_1 = DoubleConvBlock(2 * num_channels_start, 4 * num_channels_start)

        self.combined_block_2 = DoubleConvBlock(4 * num_channels_start, 8 * num_channels_start)

        # self.dropout = nn.Dropout(p=dropout_p, inplace=False)

        fc_size = 8 * num_channels_start * 6 * 6
        self.fully_connected = nn.Linear(in_features=fc_size, out_features=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, image, tof):
        # Camera branch
        x = self.conv_camera_layer_1(image)
        x = self.bn_camera_1(x)
        x = self.relu_camera_1(x)

        x = self.camera_pool_1(x)

        x = self.camera_block_1(x)

        # ToF branch
        y = self.tof_layer_1(tof)
        y = self.bn_tof_1(y)
        y = self.relu_tof_1(y)

        # Merge branches
        z = torch.cat((x, y), dim=1)  # dim 0 is batch samples, 1 is depth, 2-3 is spatial extent

        # Merged branches
        z = self.combined_block_1(z)
        z = self.combined_block_2(z)
        # z = self.dropout(z)

        # FC layer
        z = z.flatten(1)
        z = self.fully_connected(z)

        # Result
        gate_probability = self.sigmoid(z[:, 0, None])

        return gate_probability
