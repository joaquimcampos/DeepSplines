#!/usr/bin/env python3
'''
This script generates and saves a 2D circle or s_shape dataset
with a given number of training and validation samples.
'''

import argparse

from deepsplines.datasets import generate_save_dataset
from deepsplines.ds_utils import ArgCheck

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Generate and save twoD (2D) datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    dataset_choices = {'s_shape', 'circle'}
    parser.add_argument('dataset_name',
                        metavar='dataset_name [STR]',
                        choices=dataset_choices,
                        type=str,
                        help=f'{dataset_choices}')
    parser.add_argument('--data_dir',
                        metavar='[STR]',
                        type=str,
                        default='./data',
                        help=' ')
    parser.add_argument('--num_train_samples',
                        metavar='[INT,>0]',
                        type=ArgCheck.p_int,
                        default=1500,
                        help=' ')
    parser.add_argument('--num_valid_samples',
                        metavar='[INT,>0]',
                        type=ArgCheck.p_int,
                        default=1500,
                        help=' ')
    args = parser.parse_args()

    generate_save_dataset(args.dataset_name, args.data_dir,
                          args.num_train_samples, args.num_valid_samples)
