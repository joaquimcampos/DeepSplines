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
                                                'with APL activations.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TwoDSearchRun.add_default_args(parser, is_apl=True)
    args = parser.parse_args()

    srun = TwoDSearchRun(args) # instantiate search object
    params = srun.default_params('apl')
    params['verbose'] = True

    # print command-line arguments (for debugging bash scripts)
    cmd_args = ' '.join(sys.argv)
    print('\nCmd args : ', cmd_args, sep='\n')

    base_model_name = (f'{params["net"]}_{params["activation_type"]}_' +
                        f'S_apl_{params["S_apl"]}_hidden{params["hidden"]}_')

    # change gridsearch values as desired
    weight_decay_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    search_len = len(weight_decay_list)
    start_idx, end_idx = srun.init_indexes(params['log_dir'], search_len)


    for idx in range(start_idx, end_idx):

        srun.update_index_json(idx)
        weight_decay = weight_decay_list[idx]
        params['weight_decay'] = weight_decay
        params['model_name'] = base_model_name + '_weight_decay_{:.1E}'.format(weight_decay)

        combination_str = (f'\nsearch idx {idx}/{end_idx-1}, '
                            'weight_decay {:.1E}.'.format(weight_decay))

        params['combination_str'] = combination_str
        main_prog(copy.deepcopy(params))
