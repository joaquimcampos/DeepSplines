#!/usr/bin/env python3

"""
This script exemplifies how to use DeepSplines in a network.
"""

import torch
import torch.nn as nn
import torch.nn.fum

from models.deepspline_module import DeepSplineModule


# We first define a simple ReLU network to exemplify

class ReluNet(nn.Module):
    """
    Example ReLU network.

    - Input size:
        N x 2 (2D data points)
    - Output size of each layer:
        fc1 -> N x 20
        fc2 -> N x 30
        fc3 -> N x 1
    """

    def __init__(self):
        """ """
        super().__init__()

        # We'll group the activations in a ModuleList
        self.activations = nn.ModuleList()

        self.fc1 = nn.Linear(2, 20, bias=True)
        self.activations.append(nn.ReLU())

        self.fc2 = nn.Linear(20, 30, bias=True)
        self.activations.append(nn.ReLU())

        self.fc3 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        x = self.activations[0](self.fc1(x))
        x = self.activations[1](self.fc2(x))
        x = self.sigmoid(self.fc3(x)).squeeze(1)

        return x


# We now show how to replace the ReLUs with DeepSpline activations in
# the previous network.

# We need to inherit from DeepSplineModule. This is a wrap around nn.Module
# that contains all the DeepSpline functionality.
class DeepSplineNet(DeepSplineModule):
    """
    Example DeepSplines network.

    - Input size:
        N x 2 (2D data points)
    - Output size of each layer:
        fc1 -> N x 20
        fc2 -> N x 30
        fc3 -> N x 1
    """

    def __init__(self):
        """ """

        super().__init__()

        # list of 2-tuples ('layer_type', num_channels/neurons);
        # 'layer_type' can be 'conv' (convolutional) or 'fc' (fully-connected);
        activation_specs = []

        # we do not need biases since DeepSplines can do them
        self.fc1 = nn.Linear(2, 20, bias=False)
        activation_specs.append(('fc', 20)) # fully-connected layer, 20 neurons

        self.fc2 = nn.Linear(20, 30, bias=False)
        activation_specs.append(('fc', 30)) # fully-connected layer, 30 neurons

        self.fc3 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

        # len(activation_specs) = number of activation layers
        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        x = self.activations[0](self.fc1(x))
        x = self.activations[1](self.fc2(x))
        x = self.sigmoid(self.fc3(x)).squeeze(1)

        return x
