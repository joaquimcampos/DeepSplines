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
from scripts.get_activation_input_statistics import get_input_statistics


def plot_activations(params):
    """ """
    if params['no_plot'] and not params['savefig']:
        raise ValueError('Should plot or save figure')

    ckpt = Project.get_loaded_ckpt(params['ckpt_filename'])

    net  = Manager.build_model(ckpt['params'], 'cuda:0')
    net.load_state_dict(ckpt['model_state'], strict=True)
    net.eval()

    activations_list = net.get_deepspline_activations()
    num_activation_layers = len(activations_list)
    nrows = int(np.floor(np.sqrt(num_activation_layers)))
    ncols = num_activation_layers // nrows + 1

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
        x = activation_layer['x'][0].numpy()
        x_slopes = x[1:-1]
        activ = activation_layer['y'].numpy()

        assert activ.shape[0] % 2 == 0
        for i in range(activ.shape[0]):
            # plot half dashed and half full
            if i % 2 == 0:
                fig = plt.figure()
                ax = plt.gca()
                ax.grid()
                first_col = 'blue'
                sec_col = 'lightsteelblue'
                ax.plot(x, activ[i, :], linewidth=1.0, c=first_col)
            else:
                first_col = 'orange'
                sec_col = 'moccasin'
                ax.plot(x, activ[i, :], linewidth=1.0, linestyle='--', c=first_col)

            if params['plot_sparsity']:
                ls = matplotlib.rcParams['lines.markersize']
                non_sparse_slopes = (threshold_sparsity_mask[i, :] == True)
                ax.scatter(x_slopes[non_sparse_slopes], activ[i, 1:-1][non_sparse_slopes], s = 2 * (ls ** 2))

                sparse_slopes = (threshold_sparsity_mask[i, :] == False)
                ax.scatter(x_slopes[sparse_slopes], activ[i, 1:-1][sparse_slopes], color=sec_col, s = 2 * (ls ** 2))

            if i % 2 == 1:

                if params['yrange'] is not None:
                    y_range = params['yrange'][0:2] if (i <= 1) else params['yrange'][2:]
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
                            fig_save_title + activ_name + f'_neuron_pair_{i//2}') + '.pdf')

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
    parser.add_argument('--layer', metavar='INT,>=0', type=ArgCheck.p_int, help='Plot activations of this specific layer.')
    parser.add_argument('--yrange', metavar='FLOAT,>0', type=ArgCheck.p_float, nargs=4, help='Set y axis limit')
    parser.add_argument('--square', action='store_true', help='If axis shoul have same size')
    parser.add_argument('--plot_sparsity', action='store_true', help='Plot sparse/nonsparse knots')
    parser.add_argument('--no_plot', action='store_true', help='Only save figures')
    parser.add_argument('--title', type=str, help='to give default titles, give [--title '']')
    parser.add_argument('--fig_save_title', type=str, default='', help='title for saving figure')

    args = parser.parse_args()
    params = vars(args)

    plot_activations(params)
