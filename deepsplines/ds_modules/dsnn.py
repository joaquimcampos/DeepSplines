"""
Wrap around nn.Module with all DeepSpline functionalities.

DeepSpline networks should subclass DSModule.
"""

import torch
import torch.nn as nn
from torch import Tensor

from deepsplines.ds_modules.deepBspline import DeepBSpline
from deepsplines.ds_modules.deepBspline_explicit_linear import (
    DeepBSplineExplicitLinear)
from deepsplines.ds_modules.deepReLUspline import DeepReLUSpline


class DSModule(nn.Module):
    """
    Parent class for DeepSpline networks.
    """

    # dictionary with names and classes of deepspline modules
    deepsplines = {
        'deepBspline': DeepBSpline,
        'deepBspline_explicit_linear': DeepBSplineExplicitLinear,
        'deepReLUspline': DeepReLUSpline
    }

    def __init__(self, **kwargs):
        """ """

        super().__init__()

    @classmethod
    def is_deepspline_module(cls, module):
        """
        Returns True if module is a deepspline module, and False otherwise.

        Args:
            module (nn.Module)
        """
        for class_ in cls.deepsplines.values():
            if isinstance(module, class_):
                return True

        return False

    @property
    def device(self):
        """
        Get the network's device (torch.device).

        Returns the device of the first found parameter.
        """
        return next(self.parameters()).device

    def initialization(self, spline_init, init_type='He'):
        """
        Initializes the network weights with 'He', 'Xavier', or a
        custom gaussian initialization.

        spline_init (str):
            Function that activations are initialized as. Only used
            when init_type='He' and spline_init is 'leaky_relu' or 'relu'.
            Otherwise, 'Xavier' or 'custom_normal' inits are used.
        """
        if init_type not in ['He', 'Xavier', 'custom_normal']:
            raise ValueError(f'init_type {init_type} is invalid.')

        if init_type == 'He':
            if spline_init in ['leaky_relu', 'relu']:
                nonlinearity = spline_init
                slope_init = 0.01 if nonlinearity == 'leaky_relu' else 0.
            else:
                init_type = 'Xavier'  # overwrite init_type

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                if init_type == 'Xavier':
                    nn.init.xavier_normal_(module.weight)

                elif init_type == 'custom_normal':
                    # custom Gauss(0, 0.05) weight initialization
                    module.weight.data.normal_(0, 0.05)
                    module.bias.data.zero_()

                else:  # He initialization
                    nn.init.kaiming_normal_(module.weight,
                                            a=slope_init,
                                            mode='fan_out',
                                            nonlinearity=nonlinearity)

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    ###########################################################################
    # Parameters

    def get_num_params(self):
        """
        Returns the total number of network parameters.
        """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    def modules_deepspline(self):
        """
        Yields all deepspline modules in the network.
        """
        for module in self.modules():
            if self.is_deepspline_module(module):
                yield module

    def named_parameters_no_deepspline(self, recurse=True):
        """
        Yields all named_parameters in the network,
        excepting deepspline parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            deepspline_param = False
            # get all deepspline parameters
            for deepspline in self.deepsplines.values():
                for param_name in deepspline.parameter_names():
                    if name.endswith(param_name):
                        deepspline_param = True

            if deepspline_param is False:
                yield name, param

    def named_parameters_deepspline(self, recurse=True):
        """
        Yields all deepspline named_parameters in the network.
        """
        for name, param in self.named_parameters(recurse=recurse):
            deepspline_param = False
            # get all deepspline parameters
            for deepspline in self.deepsplines.values():
                for param_name in deepspline.parameter_names():
                    if name.endswith(param_name):
                        deepspline_param = True

            if deepspline_param is True:
                yield name, param

    def parameters_no_deepspline(self):
        """
        Yields all parameters in the network,
        excepting deepspline parameters.
        """
        for name, param in self.named_parameters_no_deepspline(recurse=True):
            yield param

    def parameters_deepspline(self):
        """
        Yields all deepspline parameters in the network.
        """
        for name, param in self.named_parameters_deepspline(recurse=True):
            yield param

    def freeze_parameters(self):
        """
        Freezes the network (no gradient computations).
        """
        for param in self.parameters():
            param.requires_grad = False

    ##########################################################################
    # Deepsplines: regularization and sparsification

    def l2sqsum_weights_biases(self):
        """
        Computes the sum of the l2 squared norm of the weights and biases
        of the network.

        Returns:
            l2sqsum (0d Tensor):
                l2sqsum = (sum(weights^2) + sum(biases^2))
        """
        l2sqsum = Tensor([0.]).to(self.device)

        for module in self.modules():
            if hasattr(module, 'weight') and \
                    isinstance(module.weight, nn.Parameter):
                l2sqsum = l2sqsum + module.weight.pow(2).sum()

            if hasattr(module, 'bias') and \
                    isinstance(module.bias, nn.Parameter):
                l2sqsum = l2sqsum + module.bias.pow(2).sum()

        return l2sqsum[0]  # 1-tap 1d tensor -> 0d tensor

    def TV2(self):
        """
        Computes the sum of the TV(2) (second-order total-variation)
        semi-norm of all deepspline activations in the network.

        Returns:
            tv2 (0d Tensor):
                tv2 = sum(TV(2))
        """
        tv2 = Tensor([0.]).to(self.device)

        for module in self.modules():
            if self.is_deepspline_module(module):
                module_tv2 = module.totalVariation(mode='additive')
                tv2 = tv2 + module_tv2.norm(p=1)

        return tv2[0]  # 1-tap 1d tensor -> 0d tensor

    def BV2(self):
        """
        Computes the sum of the BV(2) norm of all
        deepspline activations in the network.

        BV(2)(f) = TV(2)(f) + |f(0)| + |f(1)|
        This is a norm, whereas the TV(2) is a semi-norm.

        Returns:
            bv2 (0d Tensor):
                bv2 = sum(BV(2))
        """
        bv2 = Tensor([0.]).to(self.device)

        for module in self.modules():
            if self.is_deepspline_module(module):
                module_tv2 = module.totalVariation(mode='additive')
                module_bv2 = module_tv2 + module.fZerofOneAbs(mode='additive')
                bv2 = bv2 + module_bv2.norm(p=1)

        return bv2[0]  # 1-tap 1d tensor -> 0d tensor

    def lipschitz_bound(self):
        """
        Computes the lipschitz bound of the network

        The lipschitz bound associated with C is:
        ||f_net(x_1) - f_net(x_2)||_1 <= C ||x_1 - x_2||_1,
        for all x_1, x_2 \\in R^{N_0}.

        For l \\in {1, ..., L}, n \\in {1,..., N_l}:
        w_{n, l} is the vector of weights from layer l-1 to layer l, neuron n;
        s_{n, l} is the activation function of layer l, neuron n.

        C = (prod_{l=1}^{L} [max_{n,l} w_{n,l}]) * \
            (prod_{l=1}^{L} ||s_l||_{BV(2)}),
        where ||s_l||_{BV(2)} = sum_{n=1}^{N_l} {BV(2)(s_{n,l})}.

        Returns:
            lip_bound (0d Tensor):
                global lipschitz bound of the network
        """
        bv_product = Tensor([1.]).to(self.device)
        max_weights_product = Tensor([1.]).to(self.device)

        for module in self.modules():
            if self.is_deepspline_module(module):
                module_tv = module.totalVariation()
                module_fzero_fone = module.fZerofOneAbs()
                bv_product = bv_product * \
                    (module_tv.sum() + module_fzero_fone.sum())

            elif isinstance(module, nn.Linear) or \
                    isinstance(module, nn.Conv2d):
                max_weights_product = max_weights_product * \
                    module.weight.data.abs().max()

        lip_bound = max_weights_product * bv_product

        return lip_bound[0]  # 1-tap 1d tensor -> 0d tensor

    def sparsify_activations(self, knot_threshold):
        """
        Sparsifies the deepspline activations, eliminating slope
        changes (a_k) smaller than a threshold.

        Note that deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        This function sets a_k to zero if |a_k| < knot_threshold.

        Args:
            knot_threshold (non-negative float):
                If nonzero, sparsify activations by eliminating knots whose
                slope change is below this value.
        """
        if float(knot_threshold) < 0:
            raise TypeError('knot_threshold should be a positive float...')

        for module in self.modules():
            if self.is_deepspline_module(module):
                module.apply_threshold(float(knot_threshold))

    def compute_sparsity(self, knot_threshold):
        """
        Returns the sparsity of the activations, i.e. the number of
        activation knots whose slope change is below knot_threshold.

        Args:
            knot_threshold (non-negative float):
                threshold for slope change. If activations were sparsified
                with sparsify_activations(), this value should be equal
                to the knot_threshold used for sparsification.
        Returns:
            sparsity (int)
        """
        if float(knot_threshold) < 0:
            raise TypeError('knot_threshold should be a positive float...')

        sparsity = 0
        for module in self.modules():
            if self.is_deepspline_module(module):
                module_sparsity, _ = \
                    module.get_threshold_sparsity(float(knot_threshold))
                sparsity += module_sparsity.sum().item()

        return sparsity

    def get_deepspline_activations(self, knot_threshold=0.):
        """
        Get information of deepspline activations (e.g. for plotting).

        Args:
            knot_threshold (non-negative float):
                threshold for slope change. If activations were sparsified
                with sparsify_activations(), this value should be equal
                to the knot_threshold used for sparsification.
                If zero, all knots are True in sparsity_mask.
        Returns:
            activations_list (list):
                Length = number of deepspline activations.
                Each entry is a  dictionary of the form
                {'name': activation name,
                  'locations': position of the B-spline basis,
                  'coefficients': deepspline coefficients,
                  'threshold_sparsity_mask': mask indicating (non-zero) knots}
        """
        with torch.no_grad():
            activations_list = []
            for name, module in self.named_modules():

                if self.is_deepspline_module(module):
                    locations = module.grid_tensor  # (num_activations, size)
                    # (size, num_activations)
                    input = locations.transpose(0, 1).to(self.device)

                    if module.mode == 'conv':
                        input = input.unsqueeze(-1).unsqueeze(-1)  # to 4D

                    output = module(input)
                    coefficients = output.transpose(0, 1)

                    if module.mode == 'conv':
                        coefficients = \
                            coefficients.squeeze(-1).squeeze(-1)  # to 2D
                    # coefficients: (num_activations, size)

                    _, threshold_sparsity_mask = module.get_threshold_sparsity(
                        float(knot_threshold))

                    activations_list.append({
                        'name':
                        '_'.join([name, module.mode]),
                        'locations':
                        locations.clone().detach().cpu(),
                        'coefficients':
                        coefficients.clone().detach().cpu(),
                        'sparsity_mask':
                        threshold_sparsity_mask.cpu()
                    })

        return activations_list
