"""
This module implements deepReLU activation functions, which are
given by a sum of ReLUs with learnable slopes and a learnable linear term.

A linear spline activation with ReLU parameters {a_k},
linear parameters b1, b0, and knot locations {z_k} is described as:
deepspline(x) = sum_k [a_k * ReLU(x-z_k)] + (b1*x + b0)

The process of discovering knots is expensive. To eliminate the need for knot
discovery, we restrict the search space by placing the knots in a grid with
spacing T. As T goes to zero we can approximate any function in the original
search space.

A linear spline activation with parameters {a_k} and b1, b0, with knots placed
on a grid of spacing T, is described as:
deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)

Preference should be given to DeepBspline modules which use
an alternative B-spline representation for the linear spline.
(For more details, see deepBspline_base.py)
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.deepspline_base import DeepSplineBase



class DeepReLU(DeepSplineBase):
    """
    Class for DeepReLU activation functions.

    This activation is a sum of ReLUs with learnable slopes and
    a learnable linear term.
    """

    def __init__(self, bias=True, **kwargs):
        """
        Args:
            bias (bool):
                if True, learn bias in the linear term.
        """

        super().__init__(**kwargs)
        self.num_relus = self.size - 2
        self.learn_bias = bias

        relu_slopes = torch.zeros((self.num_activations, self.num_relus))

        # linear term coefficients (b0, b1)
        spline_bias = torch.zeros(self.num_activations)
        spline_weight = torch.zeros_like(spline_bias)

        leftmost_knot_loc = -self.grid.item() * (self.num_relus//2)
        loc_linspace = torch.linspace(leftmost_knot_loc, -leftmost_knot_loc,
                                    self.num_relus)

        # size: (num_activations, num_relus)
        knot_loc = loc_linspace.view(1, -1).expand(self.num_activations, -1)
        # by default, there is no knot discovery. If it is desired to
        # have knot discovery, make knot_loc an nn.Parameter.
        self.knot_loc = knot_loc # knot locations are not parameters

        if self.init == 'leaky_relu':
            spline_weight.fill_(0.01) # b1 = 0.01
            zero_knot_idx = self.num_relus // 2
            relu_slopes[:, zero_knot_idx].fill_(1.-0.01)

        elif self.init == 'relu':
            zero_knot_idx = self.num_relus // 2
            relu_slopes[:, zero_knot_idx].fill_(1.)

        else:
            raise ValueError('init should be in [leaky_relu, relu].')

        self._relu_slopes = nn.Parameter(relu_slopes) # size: (num_activations, num_relus)
        self.spline_weight = nn.Parameter(spline_weight) # size: (num_activations,)

        if self.learn_bias is True:
            self.spline_bias = nn.Parameter(spline_bias) # size: (num_activations,)
        else:
            self.spline_bias = spline_bias



    @staticmethod
    def parameter_names():
        """ Yield names of the module parameters """
        for name in ['relu_slopes', 'spline_weight', 'spline_bias']:
            yield name


    @property
    def weight(self):
        return self.spline_weight


    @property
    def bias(self):
        return self.spline_bias


    @property
    def relu_slopes(self):
        return self._relu_slopes



    def forward(self, input):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        input_size = input.size()
        x = self.reshape_forward(input)

        assert x.size(1) == self.num_activations, 'input.size(1) != num_activations.'

        # the deepRelu implementation is badly-conditioned: in order to
        # compute the output value at a location x, you need the value
        # (at x) of all the ReLUs that have knots before it.
        knot_loc_view = self.knot_loc.view(1, self.num_activations, 1, 1, self.num_relus)
        clamped_xknotdiff = (x.unsqueeze(-1) - knot_loc_view).clamp(min=0) # (x - \tau_k)_+

        relu_slopes_view = self.relu_slopes.view(1, self.num_activations, 1, 1, self.num_relus)
        out_relu = (relu_slopes_view * clamped_xknotdiff).sum(-1) # sum over ReLUs
        del clamped_xknotdiff

        b0 = self.spline_bias.view(1, -1, 1, 1)
        b1 = self.spline_weight.view(1, -1, 1, 1)

        out_linear = b0 + b1 * x
        output = out_relu + out_linear

        output = self.reshape_back(output, input_size)

        return output



    def apply_threshold(self, threshold):
        """
        Applies a threshold to the activations, eliminating the relu
        slopes smaller than a threshold.

        Args:
            threshold (float)
        """
        with torch.no_grad():
            new_relu_slopes = super().apply_threshold(threshold)
            self.relu_slopes.data = new_relu_slopes



    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, init={init}, '
            'num_relus={num_relus}, grid={grid[0]}, {bias}: {learn_bias}.')

        return s.format(**self.__dict__)
