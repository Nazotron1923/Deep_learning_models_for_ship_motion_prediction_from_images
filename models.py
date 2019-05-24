"""
Different neural network architectures for detecting the line
"""
from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .constants import INPUT_HEIGHT, INPUT_WIDTH
import math

class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_channel=3, drop_p=0.25, num_output=2):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.fc1 = nn.Linear(5376, 128)
        self.fc2 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, num_channel=3, drop_p=0.25, input_size=1600, hidden_size=128, num_layers=2, num_output=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding = 1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding = 1)
        )
        self.drop_p = drop_p
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_output)
        self.dropout1 = nn.Dropout(p=0.25)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        features = self.cnn(inputs)
        features = features.view(features.size(0), -1)
        features = features[None, :, :]
        outputs, _ = self.lstm(features)
        outputs = self.dropout1(outputs)
        outs = []    # save all predictions
        for time_step in range(outputs.size(1)):    # calculate output for each time step
            outs.append(self.fc2(self.dropout1(self.fc1(outputs[:, time_step, :]))))
        outs = torch.stack(outs, dim=1)
        outs = outs[0]
        return outs
