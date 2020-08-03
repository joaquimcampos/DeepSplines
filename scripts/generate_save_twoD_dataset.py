#!/usr/bin/env python3

import argparse
import os
import torch
from datasets import init_dataset
from ds_utils import ArgCheck


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Generate and save --num_datasets '
                                                    'twoD (2D) datasets with --num_samples samples each.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    dataset_choices = {'s_shape', 'circle'}
    parser.add_argument('dataset', metavar='dataset[STR]', choices=dataset_choices,
                        type=str, help=f'{dataset_choices}')
    parser.add_argument('--data_dir', metavar='STR', default='./data', type=str, help=' ')
    parser.add_argument('--num_train_samples', metavar='INT,>0', default=1500,
                        type=ArgCheck.p_int, help=' ')
    parser.add_argument('--num_valid_samples', metavar='INT,>0', default=1500,
                        type=ArgCheck.p_int, help=' ')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise OSError(f'Directory {args.data_dir} not found.')

    dataset_name = '_'.join([args.dataset, str(args.num_train_samples)])

    dataset_dir = os.path.join(args.data_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    params = {'dataset_name' : dataset_name, 'log_dir': dataset_dir,
                'plot_imgs' : False, 'save_imgs' : True}

    dataset = init_dataset(**params)

    print(f'\nGenerating {args.dataset} train/validation dataset...')

    for mode in ['train', 'valid']:
        num_samples = args.num_train_samples if mode == 'train' else args.num_valid_samples
        inputs, labels = dataset.generate_set(num_samples)

        if mode == 'train' and dataset.get_plot:
            dataset.save_title = 'Training set'
            dataset.plot_train_imgs(inputs, labels) # save training images

        save_dict = {'inputs': inputs, 'labels': labels}
        torch.save(save_dict, os.path.join(dataset.log_dir_model, mode + '_data.pth'))


    print('\nSaving test dataset...')
    dataset.save_title = 'Test set'

    inputs, labels = dataset.get_test_set()

    if dataset.get_plot:
        dataset.plot_test_imgs(inputs, labels) # save training images

    save_dict = {'inputs': inputs, 'labels': labels}
    torch.save(save_dict, os.path.join(dataset.log_dir_model, 'test_data.pth'))
