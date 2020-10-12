"""
"""

import torch
from torch import nn
from torch import Tensor
import numpy as np
from scipy.linalg import toeplitz
from abc import ABC, abstractproperty


class DeepSplineBase(ABC, nn.Module):
    """
    Args:
        mode : 'conv' or 'linear'
        size : number of coefficients of spline grid (odd number)
        grid : spacing of spline knots
        num_activations : number of convolutional filters (if mode='conv') or
                        number of linear units (if mode='linear')
        init : Initialization of activation
            (type: str or 3-tensor-tuple (bias, weight, coefficients))
    """
    def __init__(self, mode='conv', size=51, grid=0.1, num_activations=None,
                init='leaky_relu', device='cuda:0',
                dtype=torch.float32, **kwargs):

        if mode not in ['conv', 'linear']:
            raise ValueError('Mode should be either "conv" or "linear".')
        if size % 2 == 0:
            raise ValueError('Size should be an odd number.')
        if num_activations is None:
            raise ValueError('Need to provide num_activations...')

        super().__init__()

        self.mode = mode
        self.size = size
        self.num_activations = num_activations
        self.init = init
        self.device = device
        self.dtype = dtype
        self.grid = Tensor([grid]).to(**self.device_type)

        self.init_P()
        self.init_P_inv()


    @property
    def device_type(self):
        return dict(device=self.device, dtype=self.dtype)


    def init_P(self):
        """ initialize matrix that transforms deepRelu coefficients (b0, b1, (a))
        into b-spline coefficients (c).
        """
        L = self.size//2
        T = self.grid.item()

        A_first_row = np.zeros(self.size-2) # size: num_slopes
        A_first_row[0] = 1.
        A_first_col = np.arange(1, self.size-1)
        # construct A toeplitz matrix
        A = toeplitz(A_first_col, A_first_row)

        P_first_col = np.ones((self.size, 1))
        P_second_col = T*np.arange(-L, L+1)[:, np.newaxis]

        P_zeros = np.zeros((2, self.size-2))
        P_right = np.concatenate((P_zeros, T*A), axis=0)

        P = np.concatenate((P_first_col, P_second_col, P_right), axis=1)
        self.P = torch.from_numpy(P).to(**self.device_type)


    def init_P_inv(self):
        """ initialize matrix that transforms b-spline coefficients (c)
        into deepRelu coefficients (b0, b1, (a)).
        """
        L = self.size//2
        T = self.grid.item()

        D2_first_row = np.concatenate((np.array([1., -2., 1.]), np.zeros(self.size-3)))
        D2_first_col = np.zeros(self.size-2)
        D2_first_col[0] = 1.
        # construct second finite difference toeplitz matrix
        D2_mat = toeplitz(D2_first_col, D2_first_row)

        P_inv_first_row = np.zeros((1, self.size))
        P_inv_first_row[:, 0] = (1-L)
        P_inv_first_row[:, 1] = L

        P_inv_second_row = np.zeros((1, self.size))
        P_inv_second_row[:, 0] = -1/T
        P_inv_second_row[:, 1] = 1/T

        P_inv = np.concatenate((P_inv_first_row,
                                P_inv_second_row,
                                (1/T) * D2_mat), axis=0)
        self.P_inv = torch.from_numpy(P_inv).to(**self.device_type)


    @property
    def grid_tensor(self):
        return self.get_grid_tensor(self.size, self.grid)

    def get_grid_tensor(self, size_, grid_):
        """ Creates a 2D grid tensor of size (num_activations, size)
        with the positions of the B1 spline coefficients.
        """
        grid_arange = torch.arange(-(size_//2),
                                    (size_//2)+1).to(**self.device_type).mul(grid_)
        grid_tensor = grid_arange.expand((self.num_activations, size_))

        return grid_tensor


    def coefficients_to_deeprelu_coefficients(self, coefficients):
        """ Compute B-spline coefficients from deepReLU coefficients.
        """
        return (self.P_inv @ coefficients.unsqueeze(-1)).squeeze(-1)


    def deepRelu_coefficients_to_coefficients(self, deepRelu_coefficients):
        """ Convert deepReLU coefficients to deepBspline coefficients.
        """
        return (self.P @ deepRelu_coefficients.unsqueeze(-1)).squeeze(-1)


    def coefficients_grad_to_deepRelu_coefficients_grad(self, coefficients_grad):
        """ Compute gradient of deepReLU coefficients from
        gradient of B-spline coefficients using chain rule.
        """
        return (self.P.t() @ coefficients_grad.unsqueeze(-1)).squeeze(-1)


    @abstractproperty
    def coefficients(self):
        """ B-spline coefficients of activations """
        pass


    @abstractproperty
    def slopes(self):
        """ Slopes of activations """
        pass


    def totalVariation(self, **kwargs):
        """ Computes the total variation regularization: l1 norm of ReLU coefficients

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1
        """
        return self.slopes.norm(1, dim=1)



    def get_epsilon_sparsity(self, eps=5e-4):
        """ Computes the number of relus for which (|a_k| > eps).

        This function acts as a sanity check on the sparsification:
        after applying the threshold to the ReLU coefficients, we check that
        epsilon_sparsity = threshold_sparsity (check apply_threshold()).
        """
        sparsity_mask = ((self.slopes.abs() - eps) > 0.)
        sparsity = sparsity_mask.sum(dim=1)

        return sparsity, sparsity_mask



    def get_threshold_sparsity(self, threshold):
        """ Computes the number of activated relus (|a_k| > threshold)
        """
        slopes_abs = self.slopes.abs()
        threshold_sparsity_mask = (slopes_abs > threshold)
        threshold_sparsity = threshold_sparsity_mask.sum(dim=1)

        return threshold_sparsity, threshold_sparsity_mask



    def iterative_slopes_to_coefficients(self, slopes):
        """ Better conditioned than matrix formulation (see self.P)

        This way, if we set a slope to zero, we can do (b0,b1,a)->c->a'
        and still have the same slope being practically equal to zero.
        This might not be the case if we do: a' = L(self.P(b0,b1,a))
        """
        coefficients = self.coefficients
        coefficients[:, 2::] = 0. # first two coefficients remain the same

        for i in range(2, self.size):
            coefficients[:, i] = (coefficients[:, i-1] - coefficients[:, i-2]) + \
                                    slopes[:, i-2].mul(self.grid) + coefficients[:, i-1]

        return coefficients



    def apply_threshold(self, threshold):
        """ Applies a threshold to the activations, eliminating the slopes
        smaller than a threshold.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        This function sets a_k to zero if |a_k| < slope_threshold.

        When converting from a to c, we need to use the additional information
        that the first two B1 spline coefficients remain the same, i.e.,
        a = Lc, a -> sparsification -> a_hat, c_hat = f(a, c_{-L}, c_{-L+1}),
        This additional information allows us to specify the linear term parameters
        (b0, b1), whose information is in the nullspace of the second-finite-difference
        matrix L and, therefore, is lost when doing a = Lc.
        In other words, two sets of coefficients [c], [c'] which are related by
        a linear term, give the same ReLU coefficients [a].
        """
        with torch.no_grad():
            new_slopes = self.slopes
            threshold_sparsity, threshold_sparsity_mask = self.get_threshold_sparsity(threshold)
            new_slopes[~threshold_sparsity_mask] = 0.

            # sanity test - check that the slopes below threshold were indeed eliminated, i.e.,
            # are False in the epsilon_sparsity_mask.
            eps = 5e-4
            if threshold >= eps:
                epsilon_sparsity, epsilon_sparsity_mask = self.get_epsilon_sparsity(eps)
                assert epsilon_sparsity.sum() == threshold_sparsity.sum()
                assert torch.all(~epsilon_sparsity_mask[~threshold_sparsity_mask])

        return new_slopes



    def fZerofOne(self):
        """ Computes (f(0), f(1))"""
        zero_one_vec = torch.tensor([0, 1]).view(-1, 1).to(**self.device_type)
        zero_one_vec = zero_one_vec.expand((-1, self.num_activations))

        if self.mode == 'conv':
            zero_one_vec = zero_one_vec.unsqueeze(-1).unsqueeze(-1) # 4D

        out_vec = self.forward(zero_one_vec)
        if self.mode == 'conv':
            out_vec = out_vec.squeeze(-1).squeeze(-1) # (2, num_activations)
        assert out_vec.size() == (2, self.num_activations)

        return out_vec


    def fZerofOneAbs(self, **kwargs):
        """ Computes the lipschitz regularization: |f(0)| + |f(1)|
        """
        out_vec = self.fZerofOne()
        lipschitz_vec = out_vec.abs().sum(0)

        return lipschitz_vec
