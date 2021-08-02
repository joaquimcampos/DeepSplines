"""
This module implements deepBsplines with an added explicit linear term,
giving more flexibility to the activations (might produce better results)
in some contexts.

(For more details, see deepBspline_base.py.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsplines.ds_modules.deepBspline_base import DeepBSplineBase


class DeepBSpline(DeepBSplineBase):
    """ nn.Module for DeepBspline activation functions. """
    def __init__(self, mode, num_activations, **kwargs):
        """
        Args:
            mode (str):
                'conv' (convolutional) or 'fc' (fully-connected).
            num_activations :
                number of convolutional filters (if mode='conv');
                number of units (if mode='fc').
            **kwargs:
                see deepBspline_base.py/deepspline_base.py.
        """

        super().__init__(mode, num_activations, **kwargs)

        # tensor with locations of spline coefficients
        grid_tensor = self.grid_tensor  # size: (num_activations, size)
        coefficients = torch.zeros_like(grid_tensor)  # spline coefficients

        # The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators).

        if self.init == 'leaky_relu':
            coefficients = F.leaky_relu(grid_tensor, negative_slope=0.01)

        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)

        elif self.init == 'even_odd':
            # initalize half of the activations with an even function (abs) and
            # and the other half with an odd function (soft threshold).
            half = self.num_activations // 2
            coefficients[0:half, :] = (grid_tensor[0:half, :]).abs()
            coefficients[half::, :] = F.softshrink(grid_tensor[half::, :],
                                                   lambd=0.5)
        else:
            raise ValueError('init should be in [leaky_relu, relu, even_odd].')

        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self._coefficients_vect = nn.Parameter(
            coefficients.contiguous().view(-1))

    @property
    def coefficients_vect(self):
        """ B-spline vectorized coefficients. """
        return self._coefficients_vect

    @staticmethod
    def parameter_names():
        """ Yield names of the module parameters """
        yield 'coefficients_vect'

    def forward(self, input):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        output = super().forward(input)

        return output

    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]}.')

        return s.format(**self.__dict__)
