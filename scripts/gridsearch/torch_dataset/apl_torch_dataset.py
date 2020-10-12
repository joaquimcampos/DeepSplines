#!/usr/bin/env python3

import copy
import argparse
import sys

from main import main_prog
from ds_utils import ArgCheck
from torch_dataset_search_run import TorchDatasetSearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Gridsearch on cifar/mnist with a network '
                                                'with APL activations.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TorchDatasetSearchRun.add_default_args(parser, is_apl=True)
    args = parser.parse_args()

    srun = TorchDatasetSearchRun(args)
    params = srun.default_params('apl')
    params['verbose'] = True # for debugging

    # print command-line arguments (for debugging bash scripts)
    cmd_args = ' '.join(sys.argv)
    print('\nCmd args : ', cmd_args, sep='\n')

    base_model_name = (f'{params["net"]}_{params["activation_type"]}_'
                        f'S_apl_{params["S_apl"]}_'
                        'weight_decay_{:.1E}_'.format(params["weight_decay"]) +
                        f'lr_{params["lr"]}_aux_lr_{params["aux_lr"]}')

    # change gridsearch values as desired
    beta_list = [1e-3] # weight decay for parameters

    search_len = len(beta_list)
    start_idx, end_idx = srun.init_indexes(params['log_dir'], search_len)


    for idx in range(start_idx, end_idx):

        srun.update_index_json(idx)
        beta = beta_list[idx]
        params['beta'] = beta
        params['model_name'] = base_model_name + '_beta_{:.1E}'.format(beta)

        combination_str = (f'\nsearch idx {idx}/{end_idx-1}, '
                            'beta {:.1E}.'.format(beta))

        params['combination_str'] = combination_str
        main_prog(copy.deepcopy(params))
