#!/usr/bin/env python3

import argparse
import torch
from torch import nn
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from models.hybrid_deepspline import HybridDeepSpline
from models.deepRelu import DeepReLU
from ds_utils import ArgCheck, spline_grid_from_range


def parabola_func(x):
    """ Parabola function
    """
    return x ** 2


def parameters_deepRelu(activ):
    """ """
    for name, param in activ.named_parameters():
        deepRelu_param = False
        for param_name in activ.parameter_names():
            if name.endswith(param_name):
                deepRelu_param = True

        if deepRelu_param is True:
            yield param


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Approximate a parabola in [-1, 1] with a single activation',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--activation_type', type=str,
                    choices={'hybrid_deepspline', 'deepRelu'},
                    default='hybrid_deepspline', help=' ')
    parser.add_argument('--spline_size', metavar='INT>0', nargs='+',
                        type=ArgCheck.p_odd_int, default=41,
                        help='Number of b-spline/relu + linear coefficients.')
    parser.add_argument('--lmbda', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                    default=1e-2, help='TV(2) regularization.')
    args = parser.parse_args()

    size = args.spline_size
    grid = spline_grid_from_range(size, 1)
    init = 'relu'

    args_dict = {'mode': 'linear', 'num_activations': 1,
                'size': size, 'grid': grid, 'init': init,
                'device': 'cpu'}

    if args.activation_type == 'hybrid_deepspline':
        activ = HybridDeepSpline(**args_dict)
    elif args.activation_type == 'deepRelu':
        activ = DeepReLU(**args_dict)

    num_train_samples = 100
    train_x = torch.ones(num_train_samples, 1) * (2.5*grid) # between 0 and 1
    # train_x = torch.zeros(num_train_samples, 1).uniform_(0, 0.5)
    train_y = parabola_func(train_x)

    criterion = nn.MSELoss()

    lr = 1e-2
    optim_class = torch.optim.SGD # torch.optim.Adam
    optim_params = {'lr' : lr} #, 'momentum': 0.9, 'nesterov': True}

    if isinstance(activ, HybridDeepSpline):
        optim = optim_class(parameters_deepRelu(activ), **optim_params)
    else:
        optim = optim_class(activ.parameters(), **optim_params)

    print('Optimizer :', optim)

    print(f'\n==> Training {activ.__class__.__name__}')

    for i in range(3):

        optim.zero_grad()
        if isinstance(activ, HybridDeepSpline):
            activ.zero_grad_coefficients()
        pred = activ(train_x)
        df_loss = criterion(pred, train_y) # data fidelity
        df_loss.backward()

        if isinstance(activ, HybridDeepSpline):
            activ.update_deepRelu_grad()

        tv_bv_loss = args.lmbda * activ.totalVariation().sum()
        tv_bv_loss.backward()

        with torch.no_grad():
            print('deepRelu grad:, ', activ.deepRelu_coefficients_grad)

        optim.step()
        if isinstance(activ, HybridDeepSpline):
            activ.update_deepBspline_coefficients()

    print('coefficients :', activ.coefficients, sep='\n')
    print('deepRelu_coefficients :', activ.deepRelu_coefficients)
