""" See deepRelu.py, deepspline_Base.py, deepBspline.by.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from models.deepBspline import DeepBSpline
from models.deepRelu import DeepReLU
from models.deepspline_base import DeepSplineBase


class HybridDeepSpline(DeepSplineBase):
    """ see deepspline_base.py
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.deepBspline = DeepBSpline(**kwargs)
        self.deepRelu = DeepReLU(**kwargs)


    def zero_grad_coefficients(self):
        """ Sets gradients of B-spline coefficients to zero.

        Inspired by torch.optim.zero_grad method (see docs).
        """
        if self.deepBspline.coefficients_vect.grad is not None:
            self.deepBspline.coefficients_vect.grad.detach_()
            self.deepBspline.coefficients_vect.grad.zero_()


    def verify_deepReLU_zero_grad(self):
        """ """
        with torch.no_grad():
            if self.bias.grad is not None:
                assert torch.allclose(torch.zeros_like(self.bias.grad), self.bias.grad)
            if self.weight.grad is not None:
                assert torch.allclose(torch.zeros_like(self.weight.grad), self.weight.grad)
            if self.slopes.grad is not None:
                assert torch.allclose(torch.zeros_like(self.slopes.grad), self.slopes.grad)


    def update_deepRelu_grad(self):
        """ Updates the deepRelu coefficients gradients using the
        B-spline coefficients gradients.

        grad(a') = P^T grad(c).
        See DeepSplineBase.init_P in deepspline_base.py.
        """
        with torch.no_grad():
            self.verify_deepReLU_zero_grad()
            deepRelu_coefficients_grad = \
                self.coefficients_grad_to_deepRelu_coefficients_grad(self.coefficients_grad)

            self.bias.grad = deepRelu_coefficients_grad[:, 0].clone()
            self.weight.grad = deepRelu_coefficients_grad[:, 1].clone()
            self.slopes.grad = deepRelu_coefficients_grad[:, 2::].clone()


    def update_deepBspline_coefficients(self):
        """ Update deepBspline representation for deepReLU representation.
        """
        with torch.no_grad():
            coefficients = \
                self.deepRelu_coefficients_to_coefficients(self.deepRelu_coefficients)
            self.coefficients_vect.data = coefficients.view(-1).clone()


    @staticmethod
    def parameter_names(which='all'):
        """ Get hybrid deepspline parameters.

        In this case, only DeepReLU parameters are fed to the optimizer.
        The DeepBSpline parameters are only used to compute the gradients.
        """
        assert which in ['all', 'optimizer']
        for name in DeepReLU.parameter_names():
            yield name
        if which == 'all':
            for name in DeepBSpline.parameter_names():
                yield name

    @property
    def bias(self):
        return self.deepRelu.bias

    @property
    def weight(self):
        return self.deepRelu.weight

    @property
    def coefficients_vect(self):
        return self.deepBspline.coefficients_vect

    @property
    def coefficients(self):
        """ """
        return self.deepBspline.coefficients

    @property
    def coefficients_grad(self):
        """ B-spline coefficients gradients.
        """
        return self.deepBspline.coefficients_grad

    @property
    def slopes(self):
        """ """
        return self.deepRelu.slopes

    @property
    def deepRelu_coefficients(self):
        """ """
        return self.deepRelu.deepRelu_coefficients

    @property
    def deepRelu_coefficients_grad(self):
        return self.deepRelu.deepRelu_coefficients_grad



    def forward(self, input):
        """
        Args:
            x : 4D input
        """
        output = self.deepBspline(input)

        return output



    def extra_repr(self):
        """ repr for print(model)"""

        s = ('mode={mode}, num_activations={num_activations}, '
            'init={init}, size={size}, grid={grid[0]}.')

        return s.format(**self.__dict__)
