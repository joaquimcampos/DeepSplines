""" 2D-input-single-output fully-connected network with one or two layers """

import torch
import torch.nn as nn

from deepsplines.ds_modules import BaseModel


class TwoDNet(BaseModel):
    """
    - Input size:
        N x 2 (2D data points)
    - Output size of each layer:
        fc1 -> N x h
        ((fc2 -> N x h))
        fc_last -> N x 1
    """
    def __init__(self, num_hidden_layers=2, num_hidden_neurons=4, **params):

        super().__init__(**params)

        if not int(num_hidden_layers) < 3:
            raise ValueError('num_hidden_layers can only be 1 or 2.')

        self.num_hidden_layers = int(num_hidden_layers)
        self.num_hidden_neurons = int(num_hidden_neurons)

        bias = True
        if self.activation_type.startswith('deep'):
            bias = False

        activation_specs = []

        self.fc1 = nn.Linear(2, self.num_hidden_neurons, bias=bias)
        activation_specs.append(('fc', self.num_hidden_neurons))

        if num_hidden_layers > 1:
            self.fc2 = nn.Linear(self.num_hidden_neurons,
                                 self.num_hidden_neurons)
            activation_specs.append(('fc', self.num_hidden_neurons))

        self.fc_last = nn.Linear(self.num_hidden_neurons, 1)

        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()

    def forward(self, x):
        """ """
        x = self.activations[0](self.fc1(x))

        if self.num_hidden_layers > 1:
            x = self.activations[1](self.fc2(x))

        x = torch.sigmoid(self.fc_last(x)).squeeze(1)

        return x
