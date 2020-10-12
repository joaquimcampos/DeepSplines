""" This code implements linear splines activation functions.

A linear spline activation with parameters {a_k} and b1, b0, with knots placed
on a grid of spacing T, is described as:
deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)

The ReLU representation is not well-conditioned and leads to an exponential growth
with the number of coefficients of the computational and memory requirements for
training the network.
In this module, we use an alternative B1 spline representation for the activations.
the number of b-spline coefficients exceed the number of ReLU coefficients by 2,
such that len(a) + len((b1, b_0)) = len(c), so we have the same total amount of parameters.

The coefficients of the ReLU can be computed via:
a = Lc, where L is a second finite difference matrix.

This additional number of B1 spline coefficients (2), compared to the ReLU,
allows the unique specification of the linear term term, which is in the
nullspace of the L second finite-difference matrix.
In other words, two sets of coefficients [c], [c'] which are related by
a linear term, give the same ReLU coefficients [a].
Outside a region of interest, the activation is computed via left and right
linear extrapolations using the two leftmost and rightmost coefficients, respectively.

The regularization term applied to this function is:
TV(2)(deepsline) = ||a||_1 = ||Lc||_1

For the theoretical motivation and optimality results,
please see https://arxiv.org/abs/1802.09210.

For more information, please read the original deepspline network paper.
For any queries email: joaquim.campos@epfl.ch or harshit.gupta@epfl.ch.
Copyright 2020, Joaquim Campos & Harshit Gupta, All right reserved.
The user is free to use and edit the code.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np
from scipy.linalg import toeplitz
from abc import abstractproperty
from models.deepspline_base import DeepSplineBase


class DeepBSpline_Func(torch.autograd.Function):
    """ Autograd function to only backpropagate through the triangles that were used
    to calculate output = activation(input), for each element of the input.
    """

    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):

        # First, we clamp the input to the range [leftmost coefficient, second righmost coefficient].
        # We have to clamp, on the right, to the second righmost coefficient, so that we always have
        # a coefficient to the right of x_clamped to compute its output.
        # For the values outside the range, linearExtrapolations will add what remains
        # to compute the final output of the activation, taking into account the slopes
        # on the left and right.
        x_clamped = x.clamp(min = -(grid.item() * (size//2)),
                            max = (grid.item() * (size//2-1)))

        floored_x = torch.floor(x_clamped/grid) # left coefficient
        fracs = x_clamped/grid - floored_x # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left coefficients
        indexes=(zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()
        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)

        # linear interpolation
        activation_output = coefficients_vect[indexes+1]*fracs + \
                            coefficients_vect[indexes]*(1-fracs)

        return activation_output


    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (coefficients_vect[indexes+1] - coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1)+1, (fracs*grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1), ((1-fracs)*grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None



class DeepBSplineBase(DeepSplineBase):
    """ See deepspline_base.py
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.init_zero_knot_indexes()
        self.init_derivative_filters()


    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations).to(**self.device_type)
        self.zero_knot_indexes = (activation_arange*self.size + (self.size//2))


    def init_derivative_filters(self):
        """ Initialize D1, D2 filters.
        """
        # Derivative filters
        self.D1_filter = Tensor([-1,1]).view(1,1,2).to(**self.device_type).div(self.grid)
        self.D2_filter = Tensor([1,-2,1]).view(1,1,3).to(**self.device_type).div(self.grid)


    @abstractproperty
    def coefficients_vect_(self):
        """ B-spline vectorized coefficients of activations """
        pass


    @property
    def coefficients(self):
        """ B-spline coefficients.
        """
        return self.coefficients_vect_.view(self.num_activations, self.size)


    @property
    def coefficients_grad(self):
        """ B-spline coefficients gradients.
        """
        return self.coefficients_vect_.grad.view(self.num_activations, self.size)


    @property
    def slopes(self):
        """ Get the activation slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        # F.conv1d():
        # out(i, 1, :) = self.D2_filter(1, 1, :) *conv* coefficients(i, 1, :)
        # out.size() = (num_activations, 1, filtered_activation_size)
        # after filtering, we remove the singleton dimension
        slopes = F.conv1d(self.coefficients.unsqueeze(1), self.D2_filter).squeeze(1)

        return slopes



    def reshape_forward(self, input):
        """ """
        input_size = input.size()
        if self.mode == 'linear':
            if len(input_size) == 2:
                # one activation per conv channel
                x = input.view(*input_size, 1, 1) # transform to 4D size (N, num_units=num_activations, 1, 1)
            elif len(input_size) == 4:
                # one activation per conv output unit
                x = input.view(input_size[0], -1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f'input size is {len(input_size)}D but should be 2D or 4D...')
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

        return x


    def reshape_back(self, output, input_size):
        """ """
        if self.mode == 'linear':
            output = output.view(*input_size) # transform back to 2D size (N, num_units)

        return output


    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        input_size = input.size()
        x = self.reshape_forward(input)
        assert x.size(1) == self.num_activations, 'input.size(1) != num_activations.'

        # Linear extrapolations:
        # f(x_left) = leftmost coeff value + left_slope * (x - leftmost coeff)
        # f(x_right) = second rightmost coeff value + right_slope * (x - second rightmost coeff)
        # where the first components of the sums (leftmost/second rightmost coeff value)
        # are taken into account in DeepBspline_Func() and linearExtrapolations adds the rest.

        coefficients = self.coefficients
        leftmost_slope = (coefficients[:,1] - coefficients[:,0]).div(self.grid).view(1,-1,1,1)
        rightmost_slope = (coefficients[:,-1] - coefficients[:,-2]).div(self.grid).view(1,-1,1,1)

        # x.detach(): gradient w/ respect to x is already tracked in DeepBSpline_Func
        leftExtrapolations  = (x.detach() + self.grid*(self.size//2)).clamp(max=0) * leftmost_slope
        rightExtrapolations = (x.detach() - self.grid*(self.size//2-1)).clamp(min=0) * rightmost_slope
        # linearExtrapolations is zero for values inside range
        linearExtrapolations = leftExtrapolations + rightExtrapolations

        output = DeepBSpline_Func.apply(x, self.coefficients_vect_, self.grid, self.zero_knot_indexes, self.size) + \
                linearExtrapolations

        output = self.reshape_back(output, input_size)

        return output



    def reset_first_coefficients_grad(self):
        """ """
        first_knots_indexes = torch.cat((self.zero_knot_indexes - self.size//2,
                                    self.zero_knot_indexes - self.size//2 + 1))
        first_knots_indexes = first_knots_indexes.long()

        zeros = torch.zeros_like(first_knots_indexes).float()
        if not self.coefficients_vect_[first_knots_indexes].allclose(zeros):
            raise AssertionError('First coefficients are not zero...')

        self.coefficients_vect_.grad[first_knots_indexes] = zeros



    def apply_threshold(self, threshold):
        """ See DeepSplineBase.apply_threshold method
        """
        with torch.no_grad():
            new_slopes = super().apply_threshold(threshold)
            self.coefficients_vect_.data = \
                self.iterative_slopes_to_coefficients(new_slopes).view(-1)
