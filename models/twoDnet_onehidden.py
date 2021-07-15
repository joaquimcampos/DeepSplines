""" 2D-input-single-output network with one hidden layer """


import torch
import torch.nn as nn

from models.basemodel import BaseModel

__all__ = ['TwoDNet_OneHidden']


class TwoDNet_OneHidden(BaseModel):
    """
    Input size: N x 2 (2D data points)

    Output size of each layer:
    fc1 -> N x h
    fc2 -> N x 1
    """

    def __init__(self, hidden=2, **params):

        super().__init__(**params)
        self.hidden = hidden # number of hidden neurons

        activation_specs = []

        bias = True
        if self.activation_type.startswith('deep'):
            bias = False

        self.fc1 = nn.Linear(2, hidden, bias=bias)
        activation_specs.append(('fc', hidden))
        self.fc2 = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        x = self.activations[0](self.fc1(x))
        x = self.sigmoid(self.fc2(x)).squeeze(1)

        return x
