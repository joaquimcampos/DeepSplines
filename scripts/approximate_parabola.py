#!/usr/bin/env python3

import argparse
import torch
from torch import nn
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from models.deepBspline import DeepBSpline
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear
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
        for param_name in activ.parameter_names(which='optimizer'):
            if name.endswith(param_name):
                deepRelu_param = True

        if deepRelu_param is True:
            yield param


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Approximate a parabola in [-1, 1] with a single activation',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--activation_type', type=str,
                        choices={'deepBspline', 'deepRelu', 'deepBspline_explicit_linear'}
                        default='deepBspline_explicit_linear', help=' ')
    parser.add_argument('--spline_size', metavar='INT>0',
                        type=ArgCheck.p_odd_int, default=51,
                        help='Number of b-spline/relu + linear coefficients.')
    parser.add_argument('--lmbda', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                    default=1e-4, help='TV(2) regularization.')
    parser.add_argument('--num_epochs', metavar='INT,>0', type=ArgCheck.p_int,
                        default=1000, help='Number of epochs.')
    parser.add_argument('--lr', metavar='FLOAT,>0', type=ArgCheck.p_float, default=1e-3,
                        help=f'Learning rate for optimizer.')
    parser.add_argument('--num_train_samples', metavar='INT,>0', type=ArgCheck.p_int,
                        default=10000, help=' ')
    parser.add_argument('--init', choices=['even_odd', 'relu', 'leaky_relu', 'softplus', \
                        'zero', 'random', 'identity'], type=str, default='leaky_relu',
                        help=f'Initialize the b-spline coefficients according to this function.')
    parser.add_argument('--spline_range', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        default=1., help=f'One-sided range of deepspline coefficients.')
    parser.add_argument('--no_bias', action='store_true', help='Remove activation bias.')
    parser.add_argument('--lipschitz', action='store_true',
                        help='Add lipschitz regularization (controled by --lmbda)')
    parser.add_argument('--save_fig', action='store_true',
                        help='Save learned/gtruth plots.')
    parser.add_argument('--log_dir', metavar='STR', type=str,
                        help='Log directory for output learned/gtruth plots.')
    parser.add_argument('--device', choices=['cuda:0', 'cpu'], type=str,
                        default='cpu', help=' ')

    args = parser.parse_args()

    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise OSError('cuda not available...')

    if args.save_fig and args.log_dir is None:
        raise ValueError('--log_dir should be provided with --save_fig')

    if args.log_dir is not None and not os.path.isdir(args.log_dir):
        raise OSError('log_dir does not exist...')

    # print command-line arguments (for debugging bash scripts)
    cmd_args = ' '.join(sys.argv)
    print('\nCmd args : ', cmd_args, sep='\n')

    size = args.spline_size
    grid = spline_grid_from_range(size, args.spline_range)

    parab_range = 1
    args_dict = {'mode': 'linear', 'num_activations': 1,
                'bias': not args.no_bias,
                'size': size, 'grid': grid,
                'init': args.init, 'device': args.device}

    if args.activation_type == 'deepBspline':
        activation = DeepBSpline(**args_dict)
    elif args.activation_type == 'deepBspline_explicit_linear':
        activation = DeepBSplineExplicitLinear(**args_dict)
    elif args.activation_type == 'deepRelu':
        activation = DeepReLU(**args_dict)
    else:
        raise ValueError(f'Activation {args.activation_type} not available...')

    num_train_samples = args.num_train_samples
    num_test_samples = 10000

    train_x = torch.zeros(num_train_samples, 1).uniform_(-parab_range, parab_range)

    train_y = parabola_func(train_x)
    train_x = train_x.to(args.device)
    train_y = train_y.to(args.device)

    test_x = torch.zeros(num_test_samples, 1).uniform_(-parab_range, parab_range)
    test_y = parabola_func(test_x)
    test_x = test_x.to(args.device)
    test_y = test_y.to(args.device)

    criterion = nn.MSELoss().to(args.device)

    activ = activation.to(args.device)
    optim_class = torch.optim.Adam
    optim_params = {'lr' : args.lr} 
    optim = optim_class(activ.parameters(), **optim_params)
    print('Optimizer :', optim)

    num_epochs = args.num_epochs
    milestones = [int(6*num_epochs/10),
                int(8*num_epochs/10),
                int(9*num_epochs/10)]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones, gamma=0.1)

    print(f'\n==> Training {activ.__class__.__name__}')

    start_time = time.time()

    for i in range(num_epochs):

        optim.zero_grad()
        pred = activ(train_x)
        df_loss = criterion(pred, train_y) # data fidelity
        df_loss.backward()

        tv_bv_loss = args.lmbda * activ.totalVariation().sum()
        if args.lipschitz:
            tv_bv_loss = tv_bv_loss + args.lmbda * activ.fZerofOneAbs().sum()
        tv_bv_loss.backward()

        optim.step()
        scheduler.step()

        if i % 1000 == 0:
            loss = df_loss + tv_bv_loss
            print(f'\nepoch: {i+1}/{num_epochs}; ',
                  'loss: {:.8f}'.format(loss.item()), sep='')

            lr = [group['lr'] for group in optim.param_groups]
            print(f'scheduler: learning rate - {lr}')


    end_time = time.time()
    print('\nRun time: {:.5f}'.format(end_time - start_time))

    print(f'\n\n==>Testing {activ.__class__.__name__}')

    pred = activ(test_x) # prediction
    loss = criterion(pred, test_y) # data fidelity
    tv = activ.totalVariation().sum()
    test_mse_str = 'Test mse loss: {:.8f}'.format(loss.item())
    tv_loss_str = 'TV(2) loss: {:.8f}'.format(tv)
    latex_tv_loss_str = r'${\rm TV}^{(2)}$ ' + 'loss: {:.8f}'.format(tv)
    total_loss_str = 'Total loss: {:.8f}'.format(loss.item() + args.lmbda * tv)

    print(test_mse_str, tv_loss_str, total_loss_str, sep='\n')
    print('\n')

    # move to cpu and cast to numpy arrays
    test_x = test_x.cpu().numpy()[:, 0]
    test_y = test_y.cpu().numpy()[:, 0]
    pred = pred.detach().cpu().numpy()[:, 0]

    # plot gtruth, learned function
    idx = np.argsort(test_x) # sort data

    plt.plot(test_x[idx], test_y[idx])
    plt.plot(test_x[idx], pred[idx], '--')
    legend_list = ['gtruth', 'learned']

    plt.plot((-1, 1), (1/3, 1/3), 'k-')
    legend_list += ['best linear approximator']

    ax = plt.gca()
    ax.text(-0.35, 0.95, f'lambda: {args.lmbda}; size: {args.spline_size}', fontsize=8)
    ax.text(-0.35, 0.87, test_mse_str, fontsize=8)
    ax.text(-0.35, 0.79, latex_tv_loss_str, fontsize=8)
    ax.text(-0.35, 0.71, total_loss_str, fontsize=8)


    plt.legend(legend_list, fontsize=6)

    plt.title(f'Activation: {activ.__class__.__name__}')
    if args.save_fig:
        fig_save_str = (f'{activ.__class__.__name__}_' +
                        'lambda_{:.1E}_'.format(args.lmbda) +
                        f'size{args.spline_size}_range{args.spline_range}_' +
                        f'num_train_samples_{args.num_train_samples}_' +
                        'lr_{:.1E}_'.format(args.lr) +
                        f'num_epochs_{args.num_epochs}.png')
        plt.savefig(os.path.join(args.log_dir, fig_save_str))

    plt.show()
    plt.close()
