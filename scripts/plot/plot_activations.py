#!/usr/bin/env python3

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from project import Project
from manager import Manager
from models import *
from ds_utils import ArgCheck


def plot_activations(params):
    """ """
    if params['no_plot'] and not params['savefig']:
        raise ValueError('Should plot or save figure')

    ckpt = Project.get_loaded_ckpt(params['ckpt_filename'])

    device = ckpt['params']['device']
    if device == 'cuda:0' and not torch.cuda.is_available():
        raise OSError('cuda not available...')

    net  = Manager.build_model(ckpt['params'], device=device)
    net.load_state_dict(ckpt['model_state'], strict=True)
    net.eval()

    activations_list = net.get_deepspline_activations()
    num_activation_layers = len(activations_list)
    nrows = int(np.floor(np.sqrt(num_activation_layers)))
    ncols = num_activation_layers // nrows + 1
    num_activ_per_plot = params['num_activations_per_plot']

    print(f'\nNumber of activation layers: {len(activations_list)}.')

    if params['layer'] is not None:
        print(f'\nPlotting only layer {params["layer"]}/{len(activations_list)}.')
        activations_list = [activations_list[params['layer']-1]]

    for i, activation_layer in enumerate(activations_list):
        fig = None
        ax = None
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        activ_name = activation_layer['name']
        if params['title'] is not None:
            ax.set_title(activ_name, fontsize=20)

        threshold_sparsity_mask = activation_layer['threshold_sparsity_mask'].numpy()
        num_units, size = activation_layer['x'].size()
        print(f'--- plotting {i}th layer: {num_units} activations.')

        x = activation_layer['x'][0].numpy()
        x_slopes = x[1:-1]
        activ = activation_layer['y'].numpy()

        div = activ.shape[0] / num_activ_per_plot
        # number of plots for this activation layer
        total = int(np.ceil(div))
        # number of plots with params['num_activation_layers'] activations
        quotient = int(np.floor(div))
        # number of plots with less than params['num_activation_layers']
        remainder = activ.shape[0] - quotient * num_activ_per_plot

        for j in range(total):
            # plot half dashed and half full
            fig = plt.figure()
            ax = plt.gca()
            ax.grid()

            # start/end indexes of activations to plot in this layer
            start_k = j*num_activ_per_plot
            end_k = start_k + num_activ_per_plot
            if remainder != 0 and j >= total-1:
                end_k = start_k + remainder

            for k in range(start_k, end_k):
                ax.plot(x, activ[k, :], linewidth=1.0)

                if params['plot_sparsity']:
                    ls = matplotlib.rcParams['lines.markersize']
                    non_sparse_slopes = (threshold_sparsity_mask[k, :] == True)
                    ax.scatter(x_slopes[non_sparse_slopes], activ[k, 1:-1][non_sparse_slopes], s = 2 * (ls ** 2))

                    sparse_slopes = (threshold_sparsity_mask[k, :] == False)
                    ax.scatter(x_slopes[sparse_slopes], activ[k, 1:-1][sparse_slopes], s = 2 * (ls ** 2))

            if params['yrange'] is not None:
                y_range = params['yrange']
                ax.set_ylim([*y_range])
                if params['square']:
                    ax.set_xlim([*y_range])

            ax.set_xlabel(r"$x$", fontsize=20)
            ax.set_ylabel(r"$\sigma(x)$", fontsize=20)

            fig_save_title = ''
            if params['fig_save_title'] != '':
                fig_save_title = params['fig_save_title'] + '_'

            if params['savefig']:
                plt.savefig(os.path.join(params['output'],
                        fig_save_title + activ_name + f'_neurons_{start_k}_{end_k}') + '.pdf')

            plt.subplots_adjust(hspace = 0.4)

            if not params['no_plot']:
                plt.show()

            plt.close()



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Load parameters from checkpoint file.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ckpt_filename', metavar='CKPT_FILENAME', type=str, help='')
    parser.add_argument('--savefig', action='store_true', help='')
    parser.add_argument('--output', metavar='output folder', type=str, help='')
    parser.add_argument('--num_activations_per_plot', '-napp', metavar='INT,>=0', default=4, type=ArgCheck.p_int, help='Number of activations per plot.')
    parser.add_argument('--layer', metavar='INT,>=0', type=ArgCheck.p_int, help='Plot activations of this specific layer.')
    parser.add_argument('--yrange', metavar='FLOAT,>0', type=ArgCheck.p_float, nargs=2, help='Set y axis limit')
    parser.add_argument('--square', action='store_true', help='If axis shoul have same size')
    parser.add_argument('--plot_sparsity', action='store_true', help='Plot sparse/nonsparse knots')
    parser.add_argument('--no_plot', action='store_true', help='Only save figures')
    parser.add_argument('--title', type=str, help='to give default titles, give [--title '']')
    parser.add_argument('--fig_save_title', type=str, default='', help='title for saving figure')

    args = parser.parse_args()
    params = vars(args)

    plot_activations(params)
