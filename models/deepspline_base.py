import torch
from torch import nn
from abc import ABC, abstractproperty


class DeepSplineBase(ABC, nn.Module):
    """
    Abstract class for DeepSpline activations (deepReLU/deepBspline)

    Args:
        mode (str):
            'conv' (convolutional) or 'fc' (fully-connected).
        size (odd int):
            number of coefficients of spline grid;
            the number of knots K = size - 2.
        grid (float):
            spacing of spline knots.
        num_activations :
            number of convolutional filters (if mode='conv');
            number of units (if mode='fc').
        init (str):
            Function to initialize activations as (e.g. 'leaky_relu').
    """
    def __init__(self, mode='conv', size=51, grid=0.1, num_activations=None,
                init='leaky_relu', device='cuda:0',
                dtype=torch.float32, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
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
        self.grid = torch.Tensor([grid]).to(**self.device_type)



    @property
    def device_type(self):
        return dict(device=self.device, dtype=self.dtype)


    @abstractproperty
    def relu_slopes(self):
        """ ReLU slopes of activations """
        pass



    def totalVariation(self, **kwargs):
        """ Computes the total variation regularization: l1 norm of ReLU coefficients

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1
        """
        return self.relu_slopes.norm(1, dim=1)



    def get_epsilon_sparsity(self, eps=5e-4):
        """ Computes the number of relus for which (|a_k| > eps).

        This function acts as a sanity check on the sparsification:
        after applying the threshold to the ReLU coefficients, we check that
        epsilon_sparsity = threshold_sparsity (check apply_threshold()).
        """
        sparsity_mask = ((self.relu_slopes.abs() - eps) > 0.)
        sparsity = sparsity_mask.sum(dim=1)

        return sparsity, sparsity_mask



    def get_threshold_sparsity(self, threshold):
        """ Computes the number of activated relus (|a_k| > threshold)
        """
        relu_slopes_abs = self.relu_slopes.abs()
        threshold_sparsity_mask = (relu_slopes_abs > threshold)
        threshold_sparsity = threshold_sparsity_mask.sum(dim=1)

        return threshold_sparsity, threshold_sparsity_mask



    def apply_threshold(self, threshold):
        """ Applies a threshold to the activations, eliminating the relu
        slopes smaller than a threshold.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        This function sets a_k to zero if |a_k| < knot_threshold.

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
            new_relu_slopes = self.relu_slopes
            threshold_sparsity, threshold_sparsity_mask = self.get_threshold_sparsity(threshold)
            new_relu_slopes[~threshold_sparsity_mask] = 0.

            # sanity test - check that the relu slopes below threshold were
            # indeed eliminated, i.e., are False in the epsilon_sparsity_mask.
            eps = 5e-4
            if threshold >= eps:
                epsilon_sparsity, epsilon_sparsity_mask = self.get_epsilon_sparsity(eps)
                assert epsilon_sparsity.sum() == threshold_sparsity.sum()
                assert torch.all(~epsilon_sparsity_mask[~threshold_sparsity_mask])

        return new_relu_slopes



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
