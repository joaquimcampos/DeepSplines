#!/usr/bin/env python3

import os
import argparse
from ds_utils import json_load, json_dump


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Create a json file with the average of '
                                                'train/test results across runs.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json_file_path', metavar='json_file_path[STR]', type=str,
                        help='path to json file with the train/test runs results.')
    args = parser.parse_args()

    path_split = args.json_file_path.split('/')
    log_dir, json_file = '/'.join(path_split[:-1]), path_split[-1]
    avg_results_json = os.path.join(log_dir, '_'.join(['avg', json_file]))

    results_dict = json_load(args.json_file_path)
    num_runs = len(results_dict)
    avg_results_dict = {}

    first_run = next(iter(results_dict))
    # results dictionaries have, at most, depth = 2
    for key, val in results_dict[first_run].items():
        # (key, val) pairs from first run results
        if isinstance(val, dict):
            avg_results_dict[key] = {}
            for sub_key, sub_val in val.items():
                avg_val = 0.
                for run_dict in results_dict.values():
                    avg_val += float(run_dict[key][sub_key])
                avg_results_dict[key][sub_key] = float('{:.3f}'.format(avg_val/num_runs))
        else:
            avg_val = 0.
            for run_dict in results_dict.values():
                avg_val += float(run_dict[key])
            avg_results_dict[key] = float('{:.3f}'.format(avg_val/num_runs))

    json_dump(avg_results_dict, avg_results_json)

    print(f'=> Average results succesfully written to {avg_results_json}.')
