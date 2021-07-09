#!/usr/bin/env python3

import argparse
import os
import sys
from htv_utils import json_load, json_dump
import collections


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Merge two results json files',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('json_file1', metavar='json_file1[STR]', type=str,
                        help='First json file.')
    parser.add_argument('json_file2', metavar='json_file2[STR]', type=str,
                        help='Second json file. Merged json is saved here.')
    args = parser.parse_args()

    for json_file in [args.json_file1, args.json_file2]:
        if not os.path.isfile(json_file):
            raise OSError(f'{json_file} is not a valid file')

    dictio = {}
    for json_file in [args.json_file1, args.json_file2]:
        # merge with simple_json if no merged.json file seen, otherwise merge with merged.json
        results_dict = json_load(json_file)
        dictio = {**dictio, **results_dict}

    assert len(dictio) > 0
    sorting_key = 'test_mse'
    sorted_results = sorted(dictio.items(), key=lambda kv : kv[1][sorting_key])
    sorted_results_dict = collections.OrderedDict(sorted_results)

    merged_json_file = args.json_file2.split('.json')[0] + '_merged.json'
    json_dump(sorted_results_dict, merged_json_file)

    print(f'Merged file save as {merged_json_file}')
