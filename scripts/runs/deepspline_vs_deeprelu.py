#!/usr/bin/env python3

import os
import torch
import time
import copy
import argparse

from main import main_prog
from project import Project
from ds_utils import ArgCheck
from scripts.gridsearch.torch_dataset.torch_dataset_search_run import TorchDatasetSearchRun


if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='Deepspline vs Deeprelu',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = TorchDatasetSearchRun.add_default_args(parser)
    activation_choices = {'deepBspline', 'deepRelu'}
    parser.add_argument('activation_type', metavar='activation_type[STR]', type=str,
                        choices=activation_choices, help=f'{activation_choices}.')
    parser.add_argument('--num_runs', metavar='INT,>0', type=ArgCheck.p_int, default=10,
                        help='Number of runs.')
    parser.add_argument('--num_epochs', metavar='INT,>0', type=ArgCheck.p_int,
                        default=2, help=' ')

    args = parser.parse_args()
    srun = TorchDatasetSearchRun(args)
    params = srun.default_params(args.activation_type)
    params['num_epochs'] = args.num_epochs # override default number of epochs

    base_model_name = (f'{args.activation_type}_{params["net"]}_' +
                        f'size{params["spline_size"]}')

    start_idx, end_idx = srun.init_indexes(params['log_dir'], args.num_runs)


    for idx in range(start_idx, end_idx):

        srun.update_index_json(idx)
        params['model_name'] = base_model_name + f'_run{idx}'

        combination_str = (f'\nrun {idx}/{end_idx-1}')
        params['combination_str'] = combination_str
        params['verbose'] = True

        start_time = time.time()
        results = main_prog(copy.deepcopy(params))
        end_time = time.time()

        max_memory = torch.cuda.max_memory_allocated(device=params['device'])
        torch.cuda.reset_max_memory_cached(params['device'])

        # Log time/memory in train_results json file
        results_dict = Project.load_results_dict(params['log_dir'])

        results_dict[params['model_name']]['time'] = end_time - start_time
        results_dict[params['model_name']]['max_memory'] = max_memory

        Project.dump_results_dict(results_dict, params['log_dir'])
