# Adapted from
# https://echo-ai.readthedocs.io/en/latest/_modules/echoAI/Activation/Torch/apl.html#APL

import torch
import torch.nn as nn


class APL(nn.Module):
    """
    Implementation of APL (ADAPTIVE PIECEWISE LINEAR UNITS) unit:

        .. math::

            APL(x_i) = max(0,x) + \\sum_{s=1}^{S}{a_i^s * max(0, -x + b_i^s)}

    with trainable parameters a and b, parameter S should be set in advance.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - S: hyperparameter, number of hinges to be set in advance
        - a_apl: trainable parameter, control the slopes of the linear segments
        - b_apl: trainable parameter, determine the locations of the hinges

    References:
        - See APL paper:
        https://arxiv.org/pdf/1412.6830.pdf

    Examples:
        >>> a1 = apl(256, S = 1)
        >>> x = torch.randn(256)
        >>> x = a1(x)

    Args:
        mode : 'conv' or 'linear'
        num_activations : number of convolutional filters (if mode='conv') or
                        number of linear units (if mode='linear')
    """

    def __init__(self, mode='conv', num_activations=None,
                S_apl=5, device='cuda:0', dtype=torch.float32, **kwargs):
        """
        Initialization.
        INPUT:
            - mode: 'linear' if using one activation per unit or
                    'conv' if using one activation per channel
            - num_activations: number of activations to use
            - S_apl (int): number of additional knots (apart from the one at zero)
            - a_apl - value for initialization of parameter, which controls the slopes of the linear segments
            - b_apl - value for initialization of parameter, which determines the locations of the hinges
            a_apl, b_apl are initialized randomly by default
        """
        if mode not in ['conv', 'linear']:
            raise ValueError('Mode should be either "conv" or "linear".')
        if num_activations is None:
            raise ValueError('Need to provide num_activations...')

        super(APL, self).__init__()

        self.mode = mode
        self.num_activations = num_activations
        self.S_apl = S_apl
        self.device = device
        self.dtype = dtype

        # initialize parameters
        # start with small values of a
        self.a_apl = nn.Parameter(
            torch.randn((num_activations, S_apl)).div(10).to(**self.device_type)
        )

        self.b_apl = nn.Parameter(
            torch.randn((num_activations, S_apl)).to(**self.device_type)
        )


    @property
    def device_type(self):
        return dict(device=self.device, dtype=self.dtype)


    @staticmethod
    def parameter_names(**kwargs):
        for name in ['a_apl', 'b_apl']:
            yield name


    def forward(self, input):
        """
        Forward pass of the function
        Args:
            input : 2D/4D tensor
        """
        input_size = input.size()
        if self.mode == 'linear':
            if len(input_size) == 2:
                # one activation per conv channel
                x = input.view(*input_size, 1, 1) # transform to 4D size (N, num_units=num_activations, 1, 1)
            elif len(input_size) == 4:
                # one activation per conv output unit size (N, CxHxW, 1, 1)
                x = input.view(input_size[0], -1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f'input size is {len(input_size)}D but should be 2D or 4D...')
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

        assert x.size(1) == self.num_activations, 'input.size(1) != num_activations.'

        a_view = self.a_apl.view(1, self.num_activations, 1, 1, self.S_apl)
        b_view = self.b_apl.view(1, self.num_activations, 1, 1, self.S_apl)

        t = -x.unsqueeze(-1) + b_view # size: (N, C, H, W, S_apl)
        output = x.clamp(min=0) + (a_view * t.clamp(min=0)).sum(dim=-1)

        if self.mode == 'linear':
            output = output.view(*input_size) # transform back to 2D size (N, num_units)

        return output
