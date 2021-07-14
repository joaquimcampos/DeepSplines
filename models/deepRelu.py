""" This code creates a linear spline activation function, parameterized by a
ReLU expansion + linear term.

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

Preference should be given to Deepspline moydule which use
an alternative B-spline representation for the linear spline.
For more details, see deepspline_base.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.deepspline_base import DeepSplineBase



class DeepReLU(DeepSplineBase):
    """ See deepspline_base.py

    Args:
        bias : (flag) learn bias (default: True)
    """
    def __init__(self, bias=True, **kwargs):

        super().__init__(**kwargs)
        self.num_relus = self.size - 2
        self.learn_bias = bias

        relu_coefficients = torch.zeros((self.num_activations,
                                self.num_relus)).to(**self.device_type)
        # linear coefficients
        spline_bias = torch.zeros(self.num_activations).to(**self.device_type) # b0
        spline_weight = torch.zeros_like(spline_bias) # b1

        leftmost_knot_loc = -self.grid.item() * (self.num_relus//2)
        loc_linspace = torch.linspace(leftmost_knot_loc, -leftmost_knot_loc,
                                    self.num_relus).to(**self.device_type)

        # size: (num_activations, num_relus)
        knot_loc = loc_linspace.view(1, -1).expand(self.num_activations, -1)
        # by default, there is no knot discovery. If it is desired to
        # have knot discovery, set knot_loc as a nn.Parameter.
        self.knot_loc = knot_loc # knot locations are not parameters

        if self.init == 'leaky_relu':
            spline_weight.fill_(0.01) # b1 = 0.01
            zero_knot_idx = self.num_relus // 2
            relu_coefficients[:, zero_knot_idx].fill_(1.-0.01)

        elif self.init == 'relu':
            zero_knot_idx = self.num_relus // 2
            relu_coefficients[:, zero_knot_idx].fill_(1.)

        else:
            raise ValueError('init should be in [leaky_relu, relu].')

        self.relu_coefficients = nn.Parameter(relu_coefficients) # size: (num_activations, num_relus)
        self.spline_weight = nn.Parameter(spline_weight) # size: (num_activations,)

        if self.learn_bias is True:
            self.spline_bias = nn.Parameter(spline_bias) # size: (num_activations,)
        else:
            self.spline_bias = spline_bias



    @staticmethod
    def parameter_names(**kwargs):
        """ """
        for name in ['relu_coefficients', 'spline_weight', 'spline_bias']:
            yield name

    @property
    def weight(self):
        return self.spline_weight

    @property
    def bias(self):
        return self.spline_bias

    @property
    def slopes(self):
        return self.relu_coefficients

    @property
    def deepRelu_coefficients(self):
        return torch.cat((self.spline_bias.view(-1, 1),
                        self.spline_weight.view(-1, 1),
                        self.slopes), dim=1)

    @property
    def deepRelu_coefficients_grad(self):
        return torch.cat((self.spline_bias.grad.view(-1, 1),
                        self.spline_weight.grad.view(-1, 1),
                        self.slopes.grad), dim=1)



    def forward(self, input):
        """
        Args:
            x : 4D input
        """
        input_size = input.size()
        if self.mode == 'fc':
            if len(input_size) == 2:
                # one activation per conv channel
                x = input.view(*input_size, 1, 1) # transform to 4D size (N, num_units=num_activations, 1, 1)
            elif len(input_size) == 4:
                # one activation per conv output unit
                x = input.view(input_size[0], -1).unsqueeze(-1).unsqueeze(-1)
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

        assert x.size(1) == self.num_activations, 'input.size(1) != num_activations.'

        # the deepRelu implementation is badly-conditioned: in order to
        # compute the output value at a location x, you need the value
        # (at x) of all the ReLUs that have knots before it.
        knot_loc_view = self.knot_loc.view(1, self.num_activations, 1, 1, self.num_relus)
        clamped_xknotdiff = (x.unsqueeze(-1) - knot_loc_view).clamp(min=0) # (x - \tau_k)_+

        coefficients_view = self.slopes.view(1, self.num_activations, 1, 1, self.num_relus)
        out_relu = (coefficients_view * clamped_xknotdiff).sum(-1) # sum over ReLUs
        del clamped_xknotdiff

        b0 = self.spline_bias.view(1, -1, 1, 1)
        b1 = self.spline_weight.view(1, -1, 1, 1)

        out_linear = b0 + b1 * x
        output = out_relu + out_linear

        if self.mode == 'fc':
            return output.view(*input_size) # transform back to 2D size (N, num_units)

        return output



    def apply_threshold(self, threshold):
        """ See DeepSplineBase.apply_threshold method
        """
        with torch.no_grad():
            new_slopes = super().apply_threshold(threshold)
            self.slopes.data = new_slopes



    def extra_repr(self):
        """ repr for print(model)"""

        s = ('mode={mode}, num_activations={num_activations}, init={init}, '
            'num_relus={num_relus}, grid={grid[0]}, {bias}: {learn_bias}.')

        return s.format(**self.__dict__)
