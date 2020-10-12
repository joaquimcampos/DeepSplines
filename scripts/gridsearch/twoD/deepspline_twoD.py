#!/usr/bin/env python3

import copy
import argparse
import sys

from main import main_prog
from ds_utils import ArgCheck
from twoD_search_run import TwoDSearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Gridsearch on twoD dataset with a twoDnet '
                                                'with deep spline activations.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TwoDSearchRun.add_default_args(parser, is_deepspline=True)
    activation_choices = {'deepBspline', 'deepBspline_explicit_linear'}
    parser.add_argument('activation_type', metavar='activation_type[STR]',
            type=str, choices=activation_choices, help=f'{activation_choices}.')

    args = parser.parse_args()

    srun = TwoDSearchRun(args) # instantiate search object
    params = srun.default_params(args.activation_type)
    params['verbose'] = True

    # print command-line arguments (for debugging bash scripts)
    cmd_args = ' '.join(sys.argv)
    print('\nCmd args : ', cmd_args, sep='\n')

    lipschitz_str = 'lipschitz_' if params['lipschitz'] is True else ''
    size_str = '_'.join(str(i) for i in params["spline_size"])

    base_model_name = (f'{params["net"]}_{params["activation_type"]}_' +
                        f'hidden{params["hidden"]}_size{size_str}_' +
                        f'range{params["spline_range"]}_' +
                        f'{lipschitz_str}' + 'hyperparam_tuning')

    # change gridsearch values as desired
    lmbda_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    search_len = len(lmbda_list)
    start_idx, end_idx = srun.init_indexes(params['log_dir'], search_len)


    for idx in range(start_idx, end_idx):

        srun.update_index_json(idx)
        lmbda = lmbda_list[idx]
        params['lmbda'] = lmbda
        params['model_name'] = base_model_name + '_lmbda_{:.1E}'.format(lmbda)

        combination_str = (f'\nsearch idx {idx}/{end_idx-1}, '
                            'lmbda {:.1E}.'.format(lmbda))

        params['combination_str'] = combination_str
        main_prog(copy.deepcopy(params))
