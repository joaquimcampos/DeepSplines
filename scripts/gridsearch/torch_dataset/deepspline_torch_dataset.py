#!/usr/bin/env python3

import copy
import argparse

from main import main_prog
from ds_utils import ArgCheck
from torch_dataset_search_run import TorchDatasetSearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Gridsearch on cifar10/mnist with a network '
                                                'with deep spline activations.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TorchDatasetSearchRun.add_default_args(parser)
    activation_choices = {'deepBspline_explicit_linear',
                        'hybrid_deepspline', 'deepBspline_superposition'}
    parser.add_argument('activation_type', metavar='activation_type[STR]',
            type=str, choices=activation_choices, help=f'{activation_choices}.')

    args = parser.parse_args()

    srun = TorchDatasetSearchRun(args)
    params = srun.default_params(args.activation_type)
    params['verbose'] = True # for debugging

    lipschitz_str = '_lipschitz' if params['lipschitz'] is True else ''
    hyperparam_tuning_str = '_hyperparam_tuning' if params['hyperparam_tuning'] is True else ''
    multires_str = ''
    if params['multires_milestones'] is not None:
        multires_miles = '_'.join(str(i) for i in params["multires_milestones"])
        multires_str = f'_multires_{multires_miles}'

    size_str = '_'.join(str(i) for i in params["spline_size"])

    base_model_name = (f'{params["net"]}_size{size_str}_' +
                        f'range{params["spline_range"]}_' +
                        f'lr_{params["lr"]}_aux_lr_{params["aux_lr"]}'
                        f'{lipschitz_str}{hyperparam_tuning_str}{multires_str}')

    # change gridsearch values as desired
    lmbda_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1, 10, 50]

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
