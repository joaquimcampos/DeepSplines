#!/usr/bin/env python3
'''
This script plots the activation functions of a deepspline network.
The model is fetched from a checkpoint file (.pth) given as input.
Depending on the size of the network and the parameter
--num_activations_per_plot, this might produce a lot of plots.
To only check the activations for a given layer, set --layer [layer_idx].
Please run ./plot_activations.py --help for argument details.
'''

import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepsplines.project import Project
from deepsplines.manager import Manager
from deepsplines.ds_utils import ArgCheck


def plot_activations(args):
    """
    Args:
        args: verified arguments from arparser
    """
    ckpt, params = Project.load_ckpt_params(args.ckpt_filename)

    activ_type = params['model']['activation_type']
    if 'deep' not in activ_type:
        raise ValueError(
            f'Activations are of type {activ_type} and not deepspline.')

    if args.save_dir is not None and not os.path.isdir(args.save_dir):
        raise OSError(f'Save directory {args.save_dir} does not exist.')

    device = params['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        # TODO: Test how to load model on cpu trained on gpu
        raise OSError('cuda not available...')

    net = Manager.build_model(params, device=device)
    net.load_state_dict(ckpt['model_state'], strict=True)
    net.eval()
    # net.to(device)

    activations_list = net.get_deepspline_activations()
    num_activation_layers = len(activations_list)

    if args.layer is not None:
        if args.layer > len(num_activation_layers):
            raise ValueError(
                f'layer [{args.layer}] is greater than the total '
                f'number of activation layers [{num_activation_layers}].')

        print(f'\nPlotting only layer {args.layer}/{num_activation_layers}.')
        activations_list = [activations_list[args.layer - 1]]
    else:
        print(f'\nNumber of activation layers: {len(activations_list)}.')

    num_activ_per_plot = args.num_activations_per_plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i, activation_layer in enumerate(activations_list):

        activ_name = activation_layer['name']
        sparsity_mask = activation_layer['sparsity_mask'].numpy()
        num_units, size = activation_layer['locations'].size()
        print(f'--- plotting {i}th layer: {num_units} activations.')

        # Assumes that all activations have the same range/#coefficients
        locations = activation_layer['locations'][0].numpy()
        coefficients = activation_layer['coefficients'].numpy()

        div = coefficients.shape[0] * 1. / num_activ_per_plot
        # number of plots for this activation layer
        total = int(np.ceil(div))
        # number of plots with num_activ_per_plot activations
        quotient = int(np.floor(div))
        # number of activations in the last plot
        remainder = coefficients.shape[0] - quotient * num_activ_per_plot

        for j in range(total):
            # plot half dashed and half full
            plt.figure()
            ax = plt.gca()
            ax.grid()

            # start/end indexes of activations to plot in this layer
            start_k = j * num_activ_per_plot
            end_k = start_k + num_activ_per_plot
            if remainder != 0 and j >= total - 1:
                end_k = start_k + remainder

            for k in range(start_k, end_k):
                ax.plot(locations, coefficients[k, :], linewidth=1.0)

                if args.plot_sparsity:
                    ls = matplotlib.rcParams['lines.markersize']
                    non_sparse_relu_slopes = (sparsity_mask[k, :])
                    # relu slopes locations range from the second (idx=1) to
                    # second to last (idx=-1) B-spline coefficients
                    ax.scatter(locations[1:-1][non_sparse_relu_slopes],
                               coefficients[k, 1:-1][non_sparse_relu_slopes],
                               s=2 * (ls**2))

                    sparse_relu_slopes = (sparsity_mask[k, :] is False)
                    ax.scatter(locations[1:-1][sparse_relu_slopes],
                               coefficients[k, 1:-1][sparse_relu_slopes],
                               s=2 * (ls**2))

            x_range = ax.get_xlim()
            assert x_range[0] < 0 and x_range[1] > 0, f'x_range: {x_range}.'
            y_tmp = ax.get_ylim()
            assert y_tmp[0] < y_tmp[1], f'y_tmp: {y_tmp}.'

            y_range = x_range  # square axes by default
            if y_tmp[0] < x_range[0]:
                y_range[0] = y_tmp[0]
            if y_tmp[1] > x_range[1]:
                y_range[1] = y_tmp[1]

            ax.set_ylim([*y_range])
            ax.set_xlabel(r"$x$", fontsize=20)
            ax.set_ylabel(r"$\sigma(x)$", fontsize=20)

            title = activ_name + f'_neurons_{start_k}_{end_k}'
            ax.set_title(title, fontsize=20)

            if args.save_dir is not None:
                plt.savefig(os.path.join(args.save_dir, title + '.pdf'))

            # plt.subplots_adjust(hspace = 0.4)
            plt.show()
            plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Plots the activations of a deepspline network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'ckpt_filename',
        metavar='ckpt_filename [STR]',
        type=str,
        help='')
    parser.add_argument(
        '--save_dir',
        metavar='[STR]',
        type=str,
        help='directory for saving plots. If not given, plots are not saved.')
    parser.add_argument(
        '--num_activations_per_plot',
        '-napp',
        metavar='[INT,>=0]',
        default=4,
        type=ArgCheck.p_int,
        help='Number of activations per plot.')
    parser.add_argument(
        '--layer',
        metavar='[INT,>=0]',
        type=ArgCheck.p_int,
        help='Plot activations in this layer alone.')
    parser.add_argument(
        '--plot_sparsity',
        action='store_true',
        help='Plot sparse/nonsparse knots')

    args = parser.parse_args()

    plot_activations(args)
