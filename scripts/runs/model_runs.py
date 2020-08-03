#!/usr/bin/env python3

import copy
import argparse

from main import main_prog
from project import Project
from ds_utils import ArgCheck
from scripts.search_run import SearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Train/Test a model --num_runs times.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = SearchRun.add_default_args(parser)
    parser.add_argument('ckpt_filename', metavar='ckpt_filename[STR]', type=str,
                    help='Checkpoint to load model parameters for training.')
    parser.add_argument('--num_runs', metavar='INT,>0', type=ArgCheck.p_int,
                        default=9, help='Number of model runs.')

    args = parser.parse_args()
    srun = SearchRun(args)

    # get flattened checkpoint params
    _, params = Project.load_ckpt_params(args.ckpt_filename)
    base_model_name = params['model_name']
    # set params['log_dir'] = args.log_dir, params['resume'] = args.resume
    params = {**params, **srun.default_params()}

    params['test_as_valid'] = True # train on full training data
    if params['multires_milestones'] is not None:
        params['reset_multires'] = True
    if params['dataset_name'] == 'cifar10':
        params['valid_log_step'] = 391
    elif params['dataset_name'] == 'mnist':
        params['valid_log_step'] = 938

    start_idx, end_idx = srun.init_indexes(params['log_dir'], args.num_runs)


    for idx in range(start_idx, end_idx):

        srun.update_index_json(idx)
        params['model_name'] = base_model_name + f'_run{idx}'

        combination_str = (f'\nrun {idx}/{end_idx-1}')
        params['combination_str'] = combination_str

        params['mode'] = 'train'
        main_prog(copy.deepcopy(params), isloaded_params=True)

        # test model
        params['mode'] = 'test'
        main_prog(copy.deepcopy(params))
