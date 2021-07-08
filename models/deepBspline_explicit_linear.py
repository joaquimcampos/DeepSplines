""" See deepBspline_base.py.

In practice, giving deepsplines extra flexibility --- by adding an explicit
linear term to the activation --- helps.

deepBspline_explicit_linear.py = deepBspline.py + explicit linear term
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.deepBspline_base import DeepBSplineBase



class DeepBSplineExplicitLinear(DeepBSplineBase):
    """ See deepBspline_base.py

    Args:
        bias : (flag) learn bias (default: True)
        weight: (flag) learn weight (default: True)
    """
    def __init__(self, bias=True, **kwargs):

        super().__init__(**kwargs)
        self.learn_bias = bias # flag
        # used to save state
        self.input_dict = {'bias': self.learn_bias, **kwargs}

        # tensor with locations of spline coefficients
        grid_tensor = self.grid_tensor # size: (num_activations, size)
        coefficients = torch.zeros_like(grid_tensor) # spline coefficients
        # linear coefficients
        spline_bias = torch.zeros(self.num_activations).to(**self.device_type) # b0
        spline_weight = torch.zeros_like(spline_bias) # b1


        if isinstance(self.init, tuple) and len(self.init) == 3:
            if self.init[0].size() != (self.num_activations,):
                raise ValueError('Spline bias does not have the right size...')
            if self.init[1].size() != (self.num_activations,):
                raise ValueError('Spline weight does not have the right size...')
            if self.init[2].size() != (self.num_activations, self.size):
                raise ValueError('Coefficients do not have the right size...')

            spline_bias, spline_weight, coefficients = self.init

        if self.init == 'leaky_relu':
            spline_weight.fill_(0.01) # b1 = 0.01
            coefficients = F.leaky_relu(grid_tensor, negative_slope=0.01) \
                            - (0.01 * grid_tensor)

        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)

        elif self.init == 'even_odd':
            # initalize half of the activations with an even function (abs) and
            # and the other half with an odd function (soft threshold).
            half = self.num_activations // 2
            # absolute value
            spline_weight[0:half].fill_(-1.)
            coefficients[0:half, :] = (grid_tensor[0:half, :]).abs() \
                                        - (-1. * grid_tensor[0:half, :])
            # soft threshold
            spline_weight[half::].fill_(1.) # for soft threshold
            spline_bias[half::].fill_(0.5)
            coefficients[half::, :] = F.softshrink(grid_tensor[half::, :], lambd=0.5) \
                                        - (1. * grid_tensor[half::, :] + 0.5)
        else:
            raise ValueError('init should be a 3-tensor-tuple or in [leaky_relu, relu, even_odd].')

        # Need to vectorize coefficients to perform specific operations
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1)) # size: (num_activations*size)
        self.spline_weight = nn.Parameter(spline_weight) # size: (num_activations,)

        if self.learn_bias is True:
            self.spline_bias = nn.Parameter(spline_bias) # size: (num_activations,)
        else:
            self.spline_bias = spline_bias


    @property
    def coefficients_vect_(self):
        return self.coefficients_vect


    @staticmethod
    def parameter_names(**kwargs):
        for name in ['coefficients_vect', 'spline_weight', 'spline_bias']:
            yield name


    @property
    def weight(self):
        return self.spline_weight


    @property
    def bias(self):
        return self.spline_bias


    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        input_size = input.size()
        output = super().forward(input)

        x = self.reshape_forward(input)
        b0 = self.spline_bias.view((1, -1, 1, 1))
        b1 = self.spline_weight.view((1, -1, 1, 1))

        out_linear = b0 + b1 * x
        output = output + self.reshape_back(out_linear, input_size)

        return output


    def extra_repr(self):
        """ repr for print(model)
        """
        s = ('mode={mode}, num_activations={num_activations}, init={init}, '
            'size={size}, grid={grid[0]}, bias={learn_bias}.')

        return s.format(**self.__dict__)
