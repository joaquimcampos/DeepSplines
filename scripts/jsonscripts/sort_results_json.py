#!/usr/bin/env python3

import argparse
import os
import sys
from ds_utils import json_load, json_dump
import collections


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Sort json file with train/test '
                                            'results by --key value.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json_file_path', metavar='json_file_path[STR]', type=str,
                        help='path to json file with the train/test runs results.')
    parser.add_argument('key', metavar='key[STR]', type=str,
                        help='Key to sort json file by.')
    order_choices = {'ascending', 'descending'}
    parser.add_argument('--order', metavar='STR', choices=order_choices,
            default='descending', help=f'Sorting order. Choices: {str(order_choices)}')
    args = parser.parse_args()

    results_dict = json_load(args.json_file_path)
    first = next(iter(results_dict))
    assert not isinstance(results_dict[first][args.key], dict), \
	                           'model_run[key] should be int or float.'

    sorted_results = sorted(results_dict.items(), key=lambda kv : kv[1][args.key],
                        reverse=(args.order == 'descending'))
    sorted_results_dict = collections.OrderedDict(sorted_results)

    path_split = args.json_file_path.split('/')
    log_dir, json_file = '/'.join(path_split[:-1]), path_split[-1]
    sorted_results_json = os.path.join(log_dir, f'{args.key}_sorted_' + json_file)

    json_dump(sorted_results_dict, sorted_results_json)

    print(f'=> Results sorted by {args.key} succesfully written to {sorted_results_json}.')
