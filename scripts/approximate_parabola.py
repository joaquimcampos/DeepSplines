#!/usr/bin/env python3
"""
Script to approximate a parabola using a single deepsplines
activation funcion.

Used for testing purposes.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from deepsplines.ds_modules import (DeepBSpline, DeepBSplineExplicitLinear,
                                    DeepReLUSpline)
from deepsplines.ds_utils import ArgCheck, add_date_to_filename


def parabola_func(x):
    """ Parabola function """

    return x**2


def approximate_parabola(args):
    """
    Args:
        args: verified arguments from arparser
    """
    parab_range = 1  # one-sided range of parabola function

    deepspline_params = {
        'mode': 'fc',
        'size': args.spline_size,
        'range_': args.spline_range,
        'init': args.spline_init,
        'bias': True,
        'num_activations': 1,
        'save_memory': args.save_memory
    }

    if args.activation_type == 'deepBspline':
        activation = DeepBSpline(**deepspline_params)
    elif args.activation_type == 'deepBspline_explicit_linear':
        activation = DeepBSplineExplicitLinear(**deepspline_params)
    elif args.activation_type == 'deepReLUspline':
        activation = DeepReLUSpline(**deepspline_params)
    else:
        raise ValueError(f'Activation {args.activation_type} not available...')

    activation = activation.to(args.device)

    # setup training data
    train_x = torch.zeros(args.num_train_samples,
                          1).uniform_(-parab_range, parab_range)
    train_y = parabola_func(train_x)  # values
    # move to device
    train_x = train_x.to(args.device)
    train_y = train_y.to(args.device)

    # setup testing data
    num_test_samples = 10000
    test_x = torch.zeros(num_test_samples, 1).uniform_(-parab_range,
                                                       parab_range)
    test_y = parabola_func(test_x)  # values
    # move to device
    test_x = test_x.to(args.device)
    test_y = test_y.to(args.device)

    criterion = nn.MSELoss().to(args.device)

    activ = activation.to(args.device)
    optim = torch.optim.Adam(activ.parameters(), lr=args.lr)

    num_epochs = args.num_epochs
    milestones = [
        int(6 * num_epochs / 10),
        int(8 * num_epochs / 10),
        int(9 * num_epochs / 10)
    ]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                     milestones,
                                                     gamma=0.1)

    print(f'\n==> Training {activ.__class__.__name__}')

    start_time = time.time()

    # training loop
    for i in range(num_epochs):

        optim.zero_grad()
        pred = activ(train_x)
        df_loss = criterion(pred, train_y)  # data fidelity
        df_loss.backward()

        tv_bv_loss = args.lmbda * activ.totalVariation().sum()
        if args.lipschitz:
            tv_bv_loss = tv_bv_loss + args.lmbda * activ.fZerofOneAbs().sum()
        tv_bv_loss.backward()

        optim.step()
        scheduler.step()

        if i % int(num_epochs / 10) == 0:
            loss = df_loss + tv_bv_loss
            print(f'\nepoch: {i+1}/{num_epochs}; ',
                  'loss: {:.8f}'.format(loss.item()),
                  sep='')

            lr = [group['lr'] for group in optim.param_groups]
            print(f'scheduler: learning rate - {lr}')

    end_time = time.time()
    print('\nRun time: {:.5f}'.format(end_time - start_time))

    # Testing
    print(f'\n==> Start testing {activ.__class__.__name__}.\n')

    pred = activ(test_x)  # prediction
    loss = criterion(pred, test_y)  # data fidelity
    tv = activ.totalVariation().sum()

    test_mse_str = 'Test mse loss: {:.8f}'.format(loss.item())
    tv_loss_str = 'TV(2) loss: {:.8f}'.format(tv)
    total_loss_str = 'Total loss: {:.8f}'.format(loss.item() + args.lmbda * tv)

    print(test_mse_str, tv_loss_str, total_loss_str, sep='\n')

    # move to cpu and cast to numpy arrays
    test_x = test_x.cpu().numpy()[:, 0]
    test_y = test_y.cpu().numpy()[:, 0]
    pred = pred.detach().cpu().numpy()[:, 0]

    # plot gtruth, learned function
    idx = np.argsort(test_x)  # sort data

    plt.plot(test_x[idx], test_y[idx])
    plt.plot(test_x[idx], pred[idx], '--')
    legend_list = ['gtruth', 'learned']

    plt.plot((-1, 1), (1 / 3, 1 / 3), 'k-')
    legend_list += ['best linear approximator']

    plt.legend(legend_list, fontsize=6)

    plt.title(f'Activation: {activ.__class__.__name__}')
    if args.save_dir is not None:
        filename = add_date_to_filename('parabola') + '.pdf'
        plt.savefig(os.path.join(args.save_dir, filename))

    plt.show()
    plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Approximate a parabola in [-1, 1] '
        'with a single activation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # for details on the arguments, see main.py
    activation_choices = {
        'deepBspline', 'deepReLUspline', 'deepBspline_explicit_linear'
    }
    parser.add_argument('--activation_type',
                        choices=activation_choices,
                        type=str,
                        default='deepBspline_explicit_linear',
                        help=' ')
    parser.add_argument('--spline_init',
                        choices=['leaky_relu', 'relu', 'even_odd'],
                        type=str,
                        default='leaky_relu',
                        help=' ')
    parser.add_argument('--spline_size',
                        metavar='[INT>0]',
                        type=ArgCheck.p_odd_int,
                        default=31,
                        help=' ')
    parser.add_argument('--spline_range',
                        metavar='[FLOAT,>0]',
                        type=ArgCheck.p_float,
                        default=1.,
                        help=' ')
    parser.add_argument('--save_memory', action='store_true', help=' ')
    parser.add_argument('--lmbda',
                        metavar='[FLOAT,>=0]',
                        type=ArgCheck.nn_float,
                        default=1e-4,
                        help=' ')
    parser.add_argument('--lipschitz', action='store_true', help=' ')
    parser.add_argument('--num_epochs',
                        metavar='[INT,>0]',
                        type=ArgCheck.p_int,
                        default=10000,
                        help=' ')
    parser.add_argument('--lr',
                        metavar='[FLOAT,>0]',
                        type=ArgCheck.p_float,
                        default=1e-3,
                        help=' ')
    parser.add_argument('--num_train_samples',
                        metavar='[INT,>0]',
                        type=ArgCheck.p_int,
                        default=10000,
                        help=' ')

    parser.add_argument('--save_dir', metavar='[STR]', type=str)
    parser.add_argument('--device',
                        choices=['cuda:0', 'cpu'],
                        type=str,
                        default='cpu',
                        help=' ')

    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise OSError('cuda not available...')

    if (args.save_dir is not None) and (not os.path.isdir(args.save_dir)):
        print(f'\nDirectory {args.save_dir} not found. Creating it.')
        os.makedirs(args.save_dir)

    if args.save_memory is True and \
            not args.activation_type.startswith('deepBspline'):
        raise ValueError(
            '--save_memory can only be set when using deepBsplines.')

    approximate_parabola(args)
