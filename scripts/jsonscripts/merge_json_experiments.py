#!/usr/bin/env python3

import argparse
import os
import sys
from ds_utils import json_load, json_dump
import collections
from project import Project

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Merge json files from two experiments',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('log_dir1', metavar='log_dir1[STR]', type=str,
                        help='First log directory.')
    parser.add_argument('log_dir2', metavar='log_dir2[STR]', type=str,
                        help='Second log directory. Merged json is saved here.')
    args = parser.parse_args()

    for log_dir in [args.log_dir1, args.log_dir2]:
        if not os.path.isdir(log_dir):
            raise OSError(f'{log_dir} is not a valid directory')

    for mode in ['train', 'test']:
        if mode == 'train':
            base_json_filename = Project.train_results_json_filename.split('.')[0]
            sorting_key = Project.train_sorting_key
        else:
            base_json_filename = Project.test_results_json_filename.split('.')[0]
            sorting_key = Project.test_sorting_key

        dictio = {}
        for log_dir in [args.log_dir1, args.log_dir2]:
            base_json_path = os.path.join(log_dir, base_json_filename)
            json_path = None
            if os.path.isfile(base_json_path + '_merged.json'):
                json_path = base_json_path + '_merged.json'
            elif os.path.isfile(base_json_path + '.json'):
                json_path = base_json_path + '.json'
            else:
                raise ValueError(f'Did not find file {base_json_path}[.json][_merged.json] ...')

            print(f'Found file {json_path}')
            # merge with simple_json if no merged.json file seen, otherwise merge with merged.json
            results_dict = json_load(json_path)
            dictio = {**dictio, **results_dict}

        assert len(dictio) > 0
        sorted_results = sorted(dictio.items(), key=lambda kv : kv[1][sorting_key], reverse=True)
        sorted_results_dict = collections.OrderedDict(sorted_results)

        merged_json_path = os.path.join(args.log_dir2, base_json_filename + '_merged.json')
        json_dump(sorted_results_dict, merged_json_path)
