#!/usr/bin/env python3
# Based on:
# https://github.com/jiecaoyu/pytorch-nin-cifar10/blob/master/original.py

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class NINLinear(BaseModel):
    """
    input size cifar:
    N x 3 x 32 x 32
    """

    def __init__(self, **params):

        super().__init__(**params)

        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                self.init_activation(('linear', 192*(32**2)), bias=False),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                self.init_activation(('linear', 160*(32**2)), bias=False),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                self.init_activation(('linear', 96*(32**2)), bias=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                self.init_activation(('linear', 192*(16**2)), bias=False),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                self.init_activation(('linear', 192*(16**2)), bias=False),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                self.init_activation(('linear', 192*(16**2)), bias=False),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                self.init_activation(('linear', 192*(8**2)), bias=False),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                self.init_activation(('linear', 192*(8**2)), bias=False),
                nn.Conv2d(192, self.num_classes, kernel_size=1, stride=1, padding=0),
                self.init_activation(('linear', self.num_classes*(8**2)), bias=False),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

        # self.initialization(init_type='Xavier')
        self.initialization(init_type='custom_normal')
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        out = self.classifier(x)
        out = out.view(x.size(0), self.num_classes)

        return out
