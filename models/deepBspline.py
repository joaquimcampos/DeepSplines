""" See deepBspline_base.py.
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.deepBspline_base import DeepBSplineBase



class DeepBSpline(DeepBSplineBase):
    """ See deepBspline_base.py
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # tensor with locations of spline coefficients
        grid_tensor = self.grid_tensor # size: (num_activations, size)
        coefficients = torch.zeros_like(grid_tensor) # spline coefficients

        # The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators).
        if self.init == 'even_odd':
            # initalize half of the activations with an even function (abs) and
            # and the other half with an odd function (soft threshold).
            half = self.num_activations // 2
            coefficients[0:half, :] = (grid_tensor[0:half, :]).abs()
            coefficients[half::, :] = F.softshrink(grid_tensor[half::, :], lambd=0.5)

        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)

        elif self.init == 'leaky_relu':
            coefficients = F.leaky_relu(grid_tensor, negative_slope=0.01)

        elif self.init == 'softplus':
            coefficients = F.softplus(grid_tensor, beta=3, threshold=10)

        elif self.init == 'random':
            coefficients.normal_()

        elif self.init == 'identity':
            coefficients = grid_tensor.clone()

        elif self.init != 'zero':
            raise ValueError('init should be even_odd/relu/leaky_relu/softplus/'
                            'random/identity/zero]')

        # Need to vectorize coefficients to perform specific operations
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1)) # size: (num_activations*size)


    @property
    def coefficients_vect_(self):
        return self.coefficients_vect

    
    @staticmethod
    def parameter_names(**kwargs):
        yield 'coefficients_vect'


    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        output = super().forward(input)

        return output


    def extra_repr(self):
        """ repr for print(model)
        """
        s = ('mode={mode}, num_activations={num_activations}, '
            'init={init}, size={size}, grid={grid[0]}.')

        return s.format(**self.__dict__)
