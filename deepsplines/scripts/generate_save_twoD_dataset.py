#!/usr/bin/env python3

'''
This script generates and saves a 2D circle or s_shape dataset
with a given number of training and validation samples.
'''

import os
import argparse
import torch

from deepsplines.datasets import init_dataset
from deepsplines.ds_utils import ArgCheck, init_sub_dir


def generate_save_dataset(dataset_name, data_dir, num_train_samples=1500,
                    num_valid_samples=1500):
    """
    Args:
        dataset_name (str): 's_shape' or 'circle'
        data_dir (str): Data directory
        num_train_samples (int)
        num_valid_samples (int)
    """
    if not os.path.isdir(data_dir):
        print(f'\nData directory {data_dir} not found. Creating it.')
        os.makedirs(data_dir)

    dataset_dir = init_sub_dir(data_dir, dataset_name)

    params = {'dataset_name' : dataset_name, 'log_dir': dataset_dir,
                'plot_imgs' : False, 'save_imgs' : True}

    dataset = init_dataset(**params)

    print(f'\nSaving {dataset_name} dataset in {dataset_dir}')

    for mode in ['train', 'valid']:
        num_samples = num_train_samples if mode == 'train' else num_valid_samples
        inputs, labels = dataset.generate_set(num_samples)

        if mode == 'train':
            dataset.plot_train_imgs(inputs, labels) # save training images

        save_dict = {'inputs': inputs, 'labels': labels}
        torch.save(save_dict, os.path.join(dataset.log_dir_model, mode + '_data.pth'))


    inputs, labels = dataset.get_test_set()
    dataset.plot_test_imgs(inputs, labels) # save test images

    save_dict = {'inputs': inputs, 'labels': labels}
    torch.save(save_dict, os.path.join(dataset.log_dir_model, 'test_data.pth'))



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Generate and save twoD (2D) datasets.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    dataset_choices = {'s_shape', 'circle'}
    parser.add_argument('dataset_name', metavar='DATASET_NAME[STR]', choices=dataset_choices,
                        type=str, help=f'{dataset_choices}')
    parser.add_argument('--data_dir', metavar='STR', default='./data',
                        type=str, help=' ')
    parser.add_argument('--num_train_samples', metavar='INT,>0', default=1500,
                        type=ArgCheck.p_int, help=' ')
    parser.add_argument('--num_valid_samples', metavar='INT,>0', default=1500,
                        type=ArgCheck.p_int, help=' ')
    args = parser.parse_args()

    generate_save_dataset(args.dataset_name, args.data_dir, args.num_train_samples,
                        args.num_valid_samples)
