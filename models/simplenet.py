#!/usr/bin/env python3

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class SimpleNet(BaseModel):
    """
    input size cifar10:
    N x 3 x 32 x 32

    Each filtering with a filter size of 5 reduces the height and
    width of input by 4.

    Output size of each layer:
    conv1   -> (N, 6, 28, 28)
    pool2d  -> (N, 6, 14, 14)
    conv2   -> (N, 16, 10, 10)
    pool2d  -> (N, 16, 5, 5)
    reshape -> (N, 16 * 5 * 5)
    fc1     -> N x 120
    fc2     -> N x 84
    fc3     -> N x 10

    """

    def __init__(self, **params):

        super().__init__(**params)

        c1, self.c2 = 6, 16 # conv: number of channels
        l1, l2 = 120, 84  # fc: number of units

        activation_specs = [] # stores layer type ('conv'/'linear') and number of channels for each activation layer

        self.conv1 = nn.Conv2d(3, c1, 5)
        activation_specs.append(('conv', c1))
        self.pool = nn.MaxPool2d(2, 2)  # (kernel_size, stride)
        self.conv2 = nn.Conv2d(c1, self.c2, 5)
        activation_specs.append(('conv', self.c2))

        self.fc1 = nn.Linear(self.c2 * 5 * 5, l1)
        activation_specs.append(('linear', l1))
        self.fc2 = nn.Linear(l1, l2)
        activation_specs.append(('linear', l2))
        self.fc3 = nn.Linear(l2, self.num_classes)

        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        x = self.pool(self.activations[0](self.conv1(x)))
        x = self.pool(self.activations[1](self.conv2(x)))
        x = x.view(-1, self.c2 * 5 * 5)
        x = self.activations[2](self.fc1(x))
        x = self.activations[3](self.fc2(x))
        x = self.fc3(x)

        return x
