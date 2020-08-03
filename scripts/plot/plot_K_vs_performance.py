#!/usr/bin/env python3

from ds_utils import ArgCheck
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from project import Project


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description=('Plot K (deepspline size) '
                                        'vs performance.'),
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sparsified_log_dir', metavar='sparsified_log_dir[STR]',
                        type=str, help='')
    parser.add_argument('--save_fig', action='store_true', help='')
    parser.add_argument('--output', metavar='output folder', type=str, help='')
    parser.add_argument('--yrange', metavar='LIST', nargs='+', type=ArgCheck.nn_float,
                        help=' ')
    args = parser.parse_args()

    if args.sparsified_log_dir is None:
        raise ValueError('Need to provide sparsified_log_dir')

    if args.save_fig is True and args.output is None:
        raise ValueError('Need to provide output directory with --save_fig.')

    mypath = args.sparsified_log_dir
    sizes_dir = [d for d in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, d))]

    sizes = np.zeros(len(sizes_dir), dtype=np.int)
    medians = np.zeros(len(sizes_dir))

    for i, size_dir in enumerate(sizes_dir):
        size_path = os.path.join(args.sparsified_log_dir, size_dir)
        results_dict = Project.load_results_dict(size_path, mode='test')
        median_key = list(results_dict.keys())[len(results_dict)//2] # median
        medians[i] = results_dict[median_key]['test_acc']
        if size_dir.startswith('relu'):
            sizes[i] = -2
        elif size_dir.startswith('prelu'):
            sizes[i] = -1
        else:
            sizes[i] = int(size_dir[4::])

    idx = np.argsort(sizes)
    sizes, medians = sizes[idx], medians[idx]

    ticks = [str(s) for s in sizes]
    for i in [0, 1]:
        if sizes[i] == -2:
            sizes[i] = 1
            ticks[i] = 'ReLU'
        elif sizes[i] == -1:
            sizes[i] = 2
            ticks[i] = 'PReLU'

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ## K vs performance
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel(r"Number of spline coefficients", fontsize=12)
    ax.set_ylabel(r"Test accuracy ($\%$)", fontsize=12)

    ax.grid(True)
    ax.plot(sizes, medians, '--o', linewidth=1.0)

    ax.set_xscale('log')
    ax.set_xticks(sizes)
    plt.minorticks_off()
    ax.set_xticklabels(ticks)
    ax.tick_params(labelsize=8)
    if args.yrange is not None:
        assert isinstance(args.yrange, list)
        ax.set_ylim(args.yrange)

    if args.save_fig:
        plt.savefig(os.path.join(args.output, 'performance_vs_K') + '.pdf')

    plt.show()
    plt.close()
