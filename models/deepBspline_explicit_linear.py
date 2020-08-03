import torch
from torch import nn
from torch.nn import functional as F
from models.deepBspline_base import DeepBSplineBase



class DeepBSplineExplicitLinear(DeepBSplineBase):
    """ See deepBspline_base.py

    Args:
        bias : (flag) learn bias (default: True)
        weight: (flag) learn weight (default: True)
    """
    def __init__(self, bias=True, **kwargs):

        super().__init__(**kwargs)
        self.learn_bias = bias # flag
        # used to save state for increase_resolution()
        self.input_dict = {'bias': self.learn_bias, **kwargs}

        # tensor with locations of spline coefficients
        grid_tensor = self.grid_tensor # size: (num_activations, size)
        coefficients = torch.zeros_like(grid_tensor) # spline coefficients
        # linear coefficients
        spline_bias = torch.zeros(self.num_activations).to(**self.device_type) # b0
        spline_weight = torch.zeros_like(spline_bias) # b1


        if isinstance(self.init, tuple) and len(self.init) == 3:
            if self.init[0].size() != (self.num_activations,):
                raise ValueError('Spline bias does not have the right size...')
            if self.init[1].size() != (self.num_activations,):
                raise ValueError('Spline weight does not have the right size...')
            if self.init[2].size() != (self.num_activations, self.size):
                raise ValueError('Coefficients do not have the right size...')

            spline_bias, spline_weight, coefficients = self.init

        elif self.init == 'even_odd':
            # initalize half of the activations with an even function (abs) and
            # and the other half with an odd function (soft threshold).
            half = self.num_activations // 2
            # absolute value
            spline_weight[0:half].fill_(-1.)
            coefficients[0:half, :] = (grid_tensor[0:half, :]).abs() \
                                        - (-1. * grid_tensor[0:half, :])
            # soft threshold
            spline_weight[half::].fill_(1.) # for soft threshold
            spline_bias[half::].fill_(0.5)
            coefficients[half::, :] = F.softshrink(grid_tensor[half::, :], lambd=0.5) \
                                        - (1. * grid_tensor[half::, :] + 0.5)

        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)

        elif self.init == 'leaky_relu':
            spline_weight.fill_(0.01) # b1 = 0.01
            coefficients = F.leaky_relu(grid_tensor, negative_slope=0.01) \
                            - (0.01 * grid_tensor)

        elif self.init == 'random':
            coefficients.normal_()
            spline_weight.normal_()

        elif self.init == 'identity':
            spline_weight.fill_(1.)

        elif self.init != 'zero':
            raise ValueError('init should be a 3-tensor-tuple or '
                    'even_odd/relu/leaky_relu/random/identity/zero...')

        # Need to vectorize coefficients to perform specific operations
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1)) # size: (num_activations*size)
        self.spline_weight = nn.Parameter(spline_weight) # size: (num_activations,)

        if self.learn_bias is True:
            self.spline_bias = nn.Parameter(spline_bias) # size: (num_activations,)
        else:
            self.spline_bias = spline_bias



    @staticmethod
    def parameter_names(**kwargs):
        for name in ['coefficients_vect', 'spline_weight', 'spline_bias']:
            yield name

    @property
    def weight(self):
        return self.spline_weight

    @property
    def bias(self):
        return self.spline_bias



    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        output = super().forward(input)

        view_size = (1, -1) if len(input.size()) == 2 else (1, -1, 1, 1)
        b0 = self.spline_bias.view(view_size)
        b1 = self.spline_weight.view(view_size)

        out_linear = b0 + b1 * input
        output = output + out_linear

        return output



    def increase_resolution(self, order=2):
        """ Refine the activation by increasing the number of coefficients
        by an order given in the arguments.

        Initialize the new coefficients according to the activation values.
        """
        new_size = order * (self.size - 1) + 1 # e.g. 101 -> 201
        new_grid = self.grid.div(order)
        # size: (new_size, num_activations)
        new_grid_tensor = self.get_grid_tensor(new_size, new_grid).transpose(0, 1)

        if self.mode == 'conv':
            in_tensor = new_grid_tensor.unsqueeze(-1).unsqueeze(-1) # 4D

        with torch.no_grad():
            out_tensor = self.forward(in_tensor)
            if self.mode == 'conv':
                out_tensor = out_tensor.squeeze(-1).squeeze(-1) # (2, num_activations)

            # remove explicit linear part
            b0 = self.spline_bias.view(1, -1)
            b1 = self.spline_weight.view(1, -1)
            new_coefficients = out_tensor - (b0 + b1 * new_grid_tensor)

            new_coefficients = new_coefficients.transpose(0, 1)
            assert new_coefficients.size() == (self.num_activations, new_size)

        tuple_init = (self.spline_bias.data, self.spline_weight.data,
                        new_coefficients)

        new_input_dict = {'size': new_size, 'grid': new_grid.item(),
                        'init': tuple_init, **self.input_dict}

        self.__init__(**new_input_dict)



    def extra_repr(self):
        """ repr for print(model)
        """
        s = ('mode={mode}, num_activations={num_activations}, init={init}, '
            'size={size}, grid={grid[0]}, bias={learn_bias}.')

        return s.format(**self.__dict__)
