#!/usr/bin/env python3

import copy
import argparse

from main import main_prog
from ds_utils import ArgCheck
from twoD_search_run import TwoDSearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Gridsearch on twoD dataset with a twoDnet '
                                                'with standard activations.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TwoDSearchRun.add_default_args(parser, is_standard=True)
    activation_choices = {'relu', 'leaky_relu', 'prelu'}
    parser.add_argument('activation_type', metavar='activation_type[STR]',
            type=str, choices=activation_choices, help=f'{activation_choices}.')

    args = parser.parse_args()
    srun = TwoDSearchRun(args) # instantiate search object
    params = srun.default_params(args.activation_type)
    params['verbose'] = True

    base_model_name = ('{}_hidden{:d}_{}'.format(params['net'],
                        params['hidden'], params['activation_type']))

    # change gridsearch values as desired
    weight_decay_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
                        1e-3, 1e-2]

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
