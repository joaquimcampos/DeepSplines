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



    def reshape_forward(self, input):
        """
        Reshape inputs for deepspline activation forward pass, depending on
        mode ('conv' or 'fc').
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
                raise ValueError(f'input size is {len(input_size)}D but should be 2D or 4D...')
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

        return x



    def reshape_back(self, output, input_size):
        """
        Reshape back outputs after deepspline activation forward pass,
        depending on mode ('conv' or 'fc').
        """
        if self.mode == 'fc':
            output = output.view(*input_size) # transform back to 2D size (N, num_units)

        return output



    def totalVariation(self, **kwargs):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1)



    def get_epsilon_sparsity(self, eps=1e-4):
        """
        Computes the number of relus for which |a_k| > eps.

        This function acts as a sanity check on the sparsification.
        After applying the threshold to the ReLU coefficients, we check that
        epsilon_sparsity = threshold_sparsity (check apply_threshold()).
        """
        sparsity_mask = ((self.relu_slopes.abs() - eps) > 0.)
        sparsity = sparsity_mask.sum(dim=1)

        return sparsity, sparsity_mask



    def get_threshold_sparsity(self, threshold):
        """
        Computes the number of activated relus (|a_k| > threshold)
        """
        relu_slopes_abs = self.relu_slopes.abs()
        threshold_sparsity_mask = (relu_slopes_abs > threshold)
        threshold_sparsity = threshold_sparsity_mask.sum(dim=1)

        return threshold_sparsity, threshold_sparsity_mask



    def apply_threshold(self, threshold):
        """
        Applies a threshold to the activations, eliminating the relu
        slopes smaller than a threshold.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        This function sets a_k to zero if |a_k| < knot_threshold.

        Operations performed:
        . [a] = L[c], [a] -> sparsification -> [a_hat].

        Args:
            threshold (float)
        """
        with torch.no_grad():
            new_relu_slopes = self.relu_slopes
            threshold_sparsity, threshold_sparsity_mask = self.get_threshold_sparsity(threshold)
            new_relu_slopes[~threshold_sparsity_mask] = 0.

            eps = 1e-4
            if threshold >= eps:
                # sanity test: check that the relu slopes below threshold were
                # indeed eliminated, i.e., smaller than epsilon, where
                # 0 < epsilon <= threshold.
                epsilon_sparsity, epsilon_sparsity_mask = self.get_epsilon_sparsity(eps)
                assert epsilon_sparsity.sum() == threshold_sparsity.sum()
                assert torch.all(~epsilon_sparsity_mask[~threshold_sparsity_mask])

        return new_relu_slopes



    def fZerofOneAbs(self, **kwargs):
        """
        Computes |f(0)| + |f(1)| where f is a deepspline activation.

        Required for the BV(2) regularization.
        """
        zero_one_vec = torch.tensor([0, 1]).view(-1, 1).to(**self.device_type)
        zero_one_vec = zero_one_vec.expand((-1, self.num_activations))

        if self.mode == 'conv':
            zero_one_vec = zero_one_vec.unsqueeze(-1).unsqueeze(-1) # 4D

        fzero_fone = self.forward(zero_one_vec)

        if self.mode == 'conv':
            fzero_fone = fzero_fone.squeeze(-1).squeeze(-1) # (2, num_activations)
        assert fzero_fone.size() == (2, self.num_activations)

        return fzero_fone.abs().sum(0)
