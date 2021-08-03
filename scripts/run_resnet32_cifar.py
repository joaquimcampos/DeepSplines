#!/usr/bin/env python3
'''
This script reproduces the results for ResNet32
on the CIFAR10 dataset.

See https://ieeexplore.ieee.org/document/9264754.
'''

import os
import argparse
import copy
import torch

from deepsplines.main import main_prog


def run_resnet32_cifar(args):
    """
    Args:
        args: verified arguments from arparser
    """
    if not os.path.isdir(args.log_dir):
        print(f'\nLog directory {args.log_dir} not found. Creating it.')
        os.makedirs(args.log_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.activation_type == 'deepspline':
        activation_type = 'deepBspline_explicit_linear'
    else:
        activation_type = 'relu'

    params = {
        'net': 'resnet32_cifar',
        'device': device,
        'log_dir': args.log_dir,
        'num_epochs': 300,
        'milestones': [150, 225, 262],
        'activation_type': activation_type,
        'spline_init': 'leaky_relu',
        'spline_size': 51,
        'spline_range': 4,
        'save_memory': False,
        'lipschitz': False,
        'lmbda': 1e-4,
        'lr': 1e-1,
        'aux_lr': 1e-3,
        'weight_decay': 5e-4,
        'log_step': 44,  # 8 times per epoch
        'valid_log_step': -1,  # once every epoch
        'test_as_valid': True,  # print test loss at validation
        'dataset_name': 'cifar10',
        'batch_size': 128,
        'plot_imgs': False,
        'verbose': False
    }

    if activation_type == 'relu':
        params['optimizer'] = ['SGD']
    else:
        params['optimizer'] = ['SGD', 'Adam']

    params['model_name'] = f'{params["net"]}_{params["activation_type"]}_' + \
        'lambda_{:.1E}'.format(params["lmbda"])

    params['mode'] = 'train'
    main_prog(copy.deepcopy(params))

    # params['mode'] = 'test'
    # main_prog(copy.deepcopy(params))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Run ResNet32 on the CIFAR10 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log_dir',
                        metavar='[STR]',
                        type=str,
                        default='./ckpt',
                        help='Model log directory.')
    parser.add_argument('--activation_type',
                        choices=['deepspline', 'relu'],
                        type=str,
                        default='deepspline',
                        help=' ')

    args = parser.parse_args()

    run_resnet32_cifar(args)
