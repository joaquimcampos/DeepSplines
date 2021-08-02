""" Simple convolutional network for MNIST. """

import torch.nn as nn

from deepsplines.ds_modules import BaseModel


class ConvNetMnist(BaseModel):
    """
    Simple convolutional network for MNIST digit classification.

    MNIST input size: N x 1 x 28 x 28.

    Network (layer type -> output size):
    conv1   -> (N, c1, 24, 24)
    pool2d  -> (N, c1, 12, 12)
    conv2   -> (N, c1, 8, 8)
    pool2d  -> (N, c1, 4, 4)
    reshape -> (N, c1 * 4 * 4)
    fc1     -> N x 10

    The convolutions are valid convolutions with filter size = 5.
    """
    def __init__(self, **params):

        super().__init__(**params)

        c1, self.c2 = 2, 2  # conv: number of channels
        activation_specs = []

        # (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, c1, 5)
        activation_specs.append(('conv', c1))
        self.pool = nn.MaxPool2d(2, 2)  # (kernel_size, stride)
        self.conv2 = nn.Conv2d(c1, self.c2, 5)
        activation_specs.append(('conv', self.c2))
        self.fc1 = nn.Linear(self.c2 * 4 * 4, 10)

        self.activations = self.init_activation_list(activation_specs)
        self.num_params = self.get_num_params()

    def forward(self, x):
        """ """
        x = self.pool(self.activations[0](self.conv1(x)))
        x = self.pool(self.activations[1](self.conv2(x)))
        x = x.view(-1, self.c2 * 4 * 4)
        x = self.fc1(x)

        return x
