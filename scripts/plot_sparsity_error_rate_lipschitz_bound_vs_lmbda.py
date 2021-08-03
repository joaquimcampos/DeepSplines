#!/usr/bin/env python3

import os
import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from deepsplines.manager import Manager
from deepsplines.project import Project
from deepsplines.ds_utils import json_load
'''
This script plots sparsity, error_rate and lipschitz_bound vs lambda.
It takes a results_json file as input which should be located
in the same folder as the logged models.
'''


def plot_lmbda_vs_y(lmbdas, save_dir, y, y_title, file_title):
    """  """
    # lmbda vs sparsity
    plt.figure()
    ax = plt.gca()
    ax.grid(True)

    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.set_ylabel(y_title, fontsize=14)

    ax.plot(lmbdas, y, '--o', linewidth=1.0)
    ax.set_xscale('log')
    ax.set_xlim([lmbdas.min() / 10, lmbdas.max() * 10])

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, file_title + '.pdf'))

    plt.show()
    plt.close()


def plot_sparsity_error_rate_lipschitz_bound_vs_lmbda(args):
    """
    Args:
        args: verified arguments from arparser
    """
    log_dir = '/'.join(args.results_json.split('/')[:-1])
    results_dict = json_load(args.results_json)

    models = results_dict.keys()
    zeros_ = np.zeros(len(models))
    lmbdas = zeros_.copy()
    error_rates = zeros_.copy()
    sparsities = zeros_.copy()
    lipschitz_bounds = zeros_.copy()

    for i, model in enumerate(models):
        # load model lmbda
        log_dir_model = os.path.join(log_dir, model)
        ckpt_filename = Project.get_ckpt_from_log_dir_model(log_dir_model)
        ckpt, params = Project.load_ckpt_params(ckpt_filename)

        net = Manager.build_model(params, device=params['device'])
        net.load_state_dict(ckpt['model_state'], strict=True)
        net.eval()

        # save info of current model
        lmbdas[i] = params['lmbda']
        sparsities[i] = net.compute_sparsity()
        error_rates[i] = 100. - ckpt['valid_acc']
        lipschitz_bounds[i] = net.lipschitz_bound()

        # # TODO: Alternative: Get info from json file
        # try:
        #     sparsities[i] = eval(results_dict[model]['sparsity'])
        #     error_rates[i] = 100. - eval(results_dict[model][acc_str])
        #     lipschitz_bounds[i] = \
        #         eval(results_dict[model]['lipschitz_bound'])
        # except KeyError:
        #     print(f'Model {model} did not log "sparsity", "{acc_str}" '
        #             'or "lipschitz_bound".')
        #     raise

    # sort according to lmbdas
    idx = np.argsort(lmbdas)
    lmbdas = lmbdas[idx]
    sparsities = sparsities[idx]
    error_rates = error_rates[idx]
    lipschitz_bounds = lipschitz_bounds[idx]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plot_func = partial(plot_lmbda_vs_y, lmbdas, args.save_dir)

    # lmbda vs sparsity
    y_title = "Number of non-sparse coefficients"
    file_title = 'sparsity_vs_lmbda'
    plot_func(sparsities, y_title, file_title)

    # lmbda vs error_rate
    y_title = r"Error$ \ $rate (\%)"
    file_title = 'error_rate_vs_lmbda'
    plot_func(error_rates, y_title, file_title)

    # lmbda vs lipschitz_bound
    y_title = r"Lipschitz$ \ $bound"
    file_title = 'lipschitz_bound_vs_lmbda'
    plot_func(np.absolute(lipschitz_bounds), y_title, file_title)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Plot sparsity, error rate and lipschitz bound '
        'vs TV2/BV2 regularization weight.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'results_json',
        metavar='results_json [STR]',
        type=str,
        help='json file with train/test results.')
    parser.add_argument(
        '--save_dir',
        metavar='[STR]',
        type=str,
        help='directory for saving plots. If not given, plots are not saved.')
    args = parser.parse_args()

    if not os.path.isfile(args.results_json):
        raise ValueError(f'File {args.results_json} does not exist.')

    if args.save_dir is not None and not os.path.isdir(args.save_dir):
        raise OSError(f'Save directory {args.save_dir} does not exist.')

    plot_sparsity_error_rate_lipschitz_bound_vs_lmbda(args)
