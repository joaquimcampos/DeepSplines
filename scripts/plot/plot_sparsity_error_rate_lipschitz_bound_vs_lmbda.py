#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from project import Project
from manager import Manager
from models import *

import collections
from ds_utils import json_load


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Plot sparsity, error rate and lipschitz '
                                        'vs TV(2) regularization weight.'),
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sparsified_log_dir', type=str, help='')
    parser.add_argument('--savefig', action='store_true', help='')
    parser.add_argument('--output', metavar='output folder', type=str, help='')
    args = parser.parse_args()

    if args.sparsified_log_dir is None:
        raise ValueError('Need to provide sparsified_log_dir')

    results_json = os.path.join(args.sparsified_log_dir, 'avg_results.json')
    results_dict = json_load(results_json)

    models = results_dict.keys()
    lmbdas = np.zeros(len(models))
    error_rates = np.zeros(len(models))
    sparsities = np.zeros(len(models))
    lipschitz_bounds = np.zeros(len(models))

    for i, model in enumerate(models):
        model_name_split = model.split('_')
        lmbda_idx = model_name_split.index('lmbda') + 1
        lmbda = float(model_name_split[lmbda_idx])

        lmbdas[i] = lmbda
        error_rates[i] = 100. - eval(results_dict[model]['valid_acc']['median'])[0]
        sparsities[i] = eval(results_dict[model]['sparsity']['median'])[0]
        lipschitz_bounds[i] = eval(results_dict[model]['lipschitz_bound']['median'])[0]

    idx = np.argsort(lmbdas)

    lmbdas = lmbdas[idx]
    error_rates = error_rates[idx]
    sparsities = sparsities[idx]
    lipschitz_bounds = lipschitz_bounds[idx]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    lmbda_formula = (16/33)*(10 ** (-4))

    ## lmbda vs sparsity
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.set_ylabel("Number of non-sparse coefficients", fontsize=14)

    ax.grid(True)

    ax.plot(lmbdas, sparsities, '--o', linewidth=1.0)
    ax.set_xscale('log')
    ax.set_xlim([lmbdas.min()/10, lmbdas.max()*10])

    if args.savefig:
        plt.savefig(os.path.join(args.output, 'sparsity_vs_lmbda') + '.pdf')

    plt.show()
    plt.close()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ## lmbda vs error_rate
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.set_ylabel(r"Error$ \ $rate (\%)", fontsize=14)

    ax.grid(True)
    ax.plot(lmbdas, error_rates, '--o', linewidth=1.0)

    ax.set_xscale('log')
    ax.set_xlim([lmbdas.min()/10, lmbdas.max()*10])

    if args.savefig:
        plt.savefig(os.path.join(args.output, 'error_rate_vs_lmbda') + '.pdf')

    plt.show()
    plt.close()


    ## lmbda vs lipschitz_bound
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.set_ylabel(r"Lipschitz$ \ $bound", fontsize=14)

    ax.grid(True)
    ax.plot(lmbdas, np.absolute(lipschitz_bounds), '--o', linewidth=1.0)

    ax.set_xscale('log')
    ax.set_xlim([lmbdas.min()/10, lmbdas.max()*10])

    if args.savefig:
        plt.savefig(os.path.join(args.output, 'lipschitz_bound_vs_lmbda') + '.pdf')

    plt.show()
    plt.close()
