#!/usr/bin/env python3

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class TwoDNet_TwoHidden(BaseModel):
    """
    input size :
    N x 2 (2d data points)

    Output size of each layer:
    fc1 -> N x h
    fc2 -> h x h
    fc2 -> h x 1
    """

    def __init__(self, hidden=2, **params):

        super().__init__(**params)
        self.hidden = hidden # number of hidden neurons

        activation_specs = []

        bias = True
        if self.activation_type in ['deepBspline', 'deepRelu']:
            bias = False

        self.fc1 = nn.Linear(2, hidden, bias=bias)
        activation_specs.append(('linear', hidden))
        self.fc2 = nn.Linear(hidden, hidden, bias=bias)
        activation_specs.append(('linear', hidden))
        self.fc3 = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        x = self.activations[0](self.fc1(x))
        x = self.activations[1](self.fc2(x))
        x = self.sigmoid(self.fc3(x)).squeeze(1)

        return x
