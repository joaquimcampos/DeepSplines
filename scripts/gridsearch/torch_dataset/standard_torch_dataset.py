#!/usr/bin/env python3

import copy
import argparse

from main import main_prog
from ds_utils import ArgCheck
from torch_dataset_search_run import TorchDatasetSearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Gridsearch on cifar10/mnist with a network '
                                                'with standard activations.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TorchDatasetSearchRun.add_default_args(parser, is_standard=True)
    activation_choices = {'relu', 'leaky_relu', 'prelu'}
    parser.add_argument('activation_type', metavar='activation_type[STR]',
            type=str, choices=activation_choices, help=f'{activation_choices}.')

    args = parser.parse_args()
    srun = TorchDatasetSearchRun(args)
    params = srun.default_params(args.activation_type)
    params['verbose'] = True

    base_model_name = (f'{params["net"]}_{params["activation_type"]}_' 
                       f'lr_{params["lr"]}')

    # change gridsearch values as desired
    weight_decay_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    search_len = len(weight_decay_list)
    start_idx, end_idx = srun.init_indexes(params['log_dir'], search_len)


    for idx in range(start_idx, end_idx):

        srun.update_index_json(idx)
        weight_decay = weight_decay_list[idx]
        params['weight_decay'] = weight_decay
        params['model_name'] = base_model_name + '_weight_decay_{:.1E}'.format(weight_decay)

        combination_str = (f'\nsearch idx {idx}/{end_idx-1}, '
                            'weight decay {:.1E}.'.format(weight_decay))

        params['combination_str'] = combination_str
        main_prog(copy.deepcopy(params))
