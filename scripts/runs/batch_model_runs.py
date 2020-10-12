#!/usr/bin/env python3

import os
import copy
import argparse
import sys
from subprocess import call

from project import Project
from ds_utils import ArgCheck


def get_recursive_dirs(path, level, base=True):
    """ """
    if base:
        global paths
        paths = []
    elif level == 0:
        paths.append(path)
        return

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not dirs:
        raise OSError(f'No directories found in {path}... '
                        'Please recheck number of levels.')

    for dir in dirs:
        dir_path = os.path.join(path, dir)
        get_recursive_dirs(dir_path, level-1, base=False)

    if base:
        return paths


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Run several (best) models, one for each gridsearched configuration.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('log_dir', metavar='log_dir[STR]',
                type=str, help='Log directory for runs.')
    parser.add_argument('gridsearch_dir', metavar='gridsearch_log_dir[STR]',
                type=str, help='Gridsearch directory.')
    parser.add_argument('num_levels', metavar='num_levels[INT,>0]',
                type=ArgCheck.p_int, help='Number of directory nesting levels.')

    args = parser.parse_args()

    in_paths = get_recursive_dirs(args.gridsearch_dir, args.num_levels)
    out_paths = [os.path.join(args.log_dir, '/'.join(dirs.split('/')[-args.num_levels:])) \
                for dirs in in_paths]

    for path in out_paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


    for i in range(len(paths)):
        _, ckpt_filename = Project.get_best_model(paths[i], mode='train')
        run_cmd = (f'python3 scripts/runs/model_runs.py {out_paths[i]} '
                    f'{ckpt_filename}')

        print('\nRun cmd:', run_cmd, sep='\n')

        call(run_cmd.split(' '))
