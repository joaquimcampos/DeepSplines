import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np
from scipy.linalg import toeplitz
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
        self.init_admm() # TODO: Remove


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


    # TODO: Remove
    def init_admm(self):
        """ Explicitly build toeplitz regularization matrix L and
        initialize tensors for ADMM
        """
        D2_filter_np = self.D2_filter.view(-1).cpu().numpy()
        # define column of L matrix
        L_first_col = np.zeros(self.size-2)
        L_first_col[0] = D2_filter_np[0]
        # define row of L matrix
        L_first_row = np.zeros(self.size)
        L_first_row[0:3] = D2_filter_np
        # construct L toeplitz matrix
        L = toeplitz(L_first_col, L_first_row)
        self.L = torch.from_numpy(L).to(**self.device_type)

        self.Id = torch.eye(self.size).to(**self.device_type)
        self.LT = self.L.t() # size: (self.size, self.size-2)
        self.LTL = self.LT @ self.L # size: (self.size, self.size)
        self.init_uvec = \
            torch.zeros((self.num_activations, self.size-2, 1)).to(**self.device_type)



    @property
    def coefficients(self):
        """ B-spline coefficients.
        """
        return self.coefficients_vect.view(self.num_activations, self.size)

    @property
    def coefficients_grad(self):
        """ B-spline coefficients gradients.
        """
        return self.coefficients_vect.grad.view(self.num_activations, self.size)

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



    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        input_size = input.size()
        if self.mode == 'linear':
            assert len(input_size) == 2, 'input to activation should be 2D (N, num_units) if mode="linear".'
            x = input.view(*input_size, 1, 1) # transform to 4D size (N, num_units=num_activations, 1, 1)
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

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

        output = DeepBSpline_Func.apply(x, self.coefficients_vect, self.grid, self.zero_knot_indexes, self.size) + \
                linearExtrapolations

        if self.mode == 'linear':
            output = output.view(*input_size) # transform back to 2D size (N, num_units)

        return output



    def reset_first_coefficients_grad(self):
        """ """
        first_knots_indexes = torch.cat((self.zero_knot_indexes - self.size//2,
                                    self.zero_knot_indexes - self.size//2 + 1))
        first_knots_indexes = first_knots_indexes.long()

        zeros = torch.zeros_like(first_knots_indexes).float()
        if not self.coefficients_vect[first_knots_indexes].allclose(zeros):
            raise AssertionError('First coefficients are not zero...')

        self.coefficients_vect.grad[first_knots_indexes] = zeros



    def apply_threshold(self, threshold):
        """ See DeepSplineBase.apply_threshold method
        """
        with torch.no_grad():
            new_slopes = super().apply_threshold(threshold)
            self.coefficients_vect.data = \
                self.iterative_slopes_to_coefficients(new_slopes).view(-1)


    # TODO: Remove
    def update_admm(self, lmbda):
        """ Update A matrix """
        self.lmbda = lmbda
        self.A = torch.inverse(self.Id + self.lmbda*self.LTL) # size: (self.size, self.size)

    # TODO: Remove
    def apply_prox(self, prox_iter):
        """
        Apply proximal operator computed by ADMM
        """
        with torch.no_grad():

            # coefficients size: (self.num_activations, self.size, 1)
            orig_coefficients = self.coefficients.unsqueeze(-1).clone().detach()
            coefficients = orig_coefficients.clone()
            uvec = self.init_uvec

            for i in range(prox_iter):
                Lc = torch.matmul(self.L, coefficients)
                assert Lc.size() == (self.num_activations, self.size-2, 1)

                # z_vec size: (self.num_activations, self.size-2, 1)
                zvec = F.softshrink(Lc + uvec, lambd=self.lmbda)
                uvec = uvec + Lc - zvec
                b = orig_coefficients + self.lmbda * torch.matmul(self.LT, zvec-uvec)
                assert b.size() == (self.num_activations, self.size, 1)

                coefficients = torch.matmul(self.A, b)
                assert coefficients.size() == (self.num_activations, self.size, 1)

            self.coefficients_vect.data = coefficients.view(-1)
