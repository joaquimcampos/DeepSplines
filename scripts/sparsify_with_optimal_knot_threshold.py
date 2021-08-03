#!/usr/bin/env python3
'''
This script sparsifies the deepspline activations in a network.

It takes a trained model from a checkpoint (.pth)
and a tolerance "acc_drop_threshold" (in (-1, 0)) corresponding to the
training accuracy drop w.r.t. the original model that is tolerated.
The script then looks for the highest knot_threshold
(higher => sparser activations) that is allowed by the specifications.
'''

import os
import sys
import argparse
import copy
import shutil
import numpy as np

from deepsplines.ds_utils import ArgCheck
from deepsplines.project import Project
from deepsplines.main import main_prog


def delete_model(log_dir, model_name, results_dict):
    # delete model directory
    log_dir_model = os.path.join(log_dir, model_name)
    assert os.path.isdir(log_dir_model)
    shutil.rmtree(log_dir_model)
    # delete model entry in train_result.json
    del results_dict[model_name]
    Project.dump_results_dict(results_dict, log_dir)


def sparsify_with_optimal_knot_threshold(args):
    """
    Args:
        args: verified arguments from arparser
    """
    ckpt, params = Project.load_ckpt_params(args.ckpt_filename, flatten=True)

    if 'deep' not in params['activation_type']:
        raise ValueError('This ckpt contains activations of type '
                         '{params["activation_type"]} and not deepsplines.')

    base_model_name = params['model_name']
    print('\n=> Compute optimal threshold/sparsity '
          f'for model: {base_model_name}.')

    # take the trained model, do one more epoch with a slope threshold
    # applied and the model frozen, and compute the train accuracy.
    params['resume'] = True
    params['num_epochs'] = ckpt['num_epochs_finished']
    params['log_step'] = None  # at every epoch
    params['valid_log_step'] = -1  # at every epoch
    params['log_dir'] = args.out_log_dir
    params['ckpt_filename'] = args.ckpt_filename

    # also log sparsity and lipschitz bound
    params['additional_info'] = ['sparsity']
    if params['lipschitz'] is True:
        params['additional_info'].append('lipschitz_bound')

    # variable initialization
    base_train_acc = 0.
    prev_model_name = None  # previous model name
    chosen_model, chosen_threshold = None, None

    # training accuracy maximum allowed percentage drop
    acc_drop_threshold = args.acc_drop_threshold
    threshold_list = np.concatenate(
        (np.zeros(1), np.arange(0.0002, 0.004,
                                0.0002), np.arange(0.004, 1, 0.05),
         np.arange(1, 3, 0.2), np.arange(3, 10, 0.5), np.arange(10, 100, 2)))

    for k in range(threshold_list.shape[0]):

        threshold = threshold_list[k]
        params['model_name'] = base_model_name + \
            '_knot_threshold_{:.4f}'.format(threshold)
        params['knot_threshold'] = threshold

        sys.stdout = open(os.devnull, "w")
        main_prog(copy.deepcopy(params), isloaded_params=True)
        sys.stdout = sys.__stdout__

        results_dict = Project.load_results_dict(args.out_log_dir)
        model_dict = results_dict[params['model_name']]

        if k == 0:
            assert np.allclose(threshold, 0)
            # TODO: Abstract from 'latest_train_acc'
            base_train_acc = model_dict['latest_train_acc']

        acc_drop = np.clip((model_dict['latest_train_acc'] - base_train_acc),
                           a_max=100,
                           a_min=-100)

        print('\nThreshold: {:.4f}'.format(threshold))
        print('Accuracy drop: {:.3f}%'.format(acc_drop))
        print('Sparsity: {:d}'.format(int(model_dict['sparsity'])))

        if acc_drop < acc_drop_threshold or model_dict['sparsity'] == 0:
            # delete current model; chosen_model is the previous one
            delete_model(args.out_log_dir, params['model_name'], results_dict)
            break
        else:
            if k > 0:
                # delete previous model
                delete_model(args.out_log_dir, prev_model_name, results_dict)

            chosen_model = {params['model_name']: model_dict}
            chosen_threshold = params['knot_threshold']
            prev_model_name = params['model_name']

    assert chosen_model is not None
    assert chosen_threshold is not None
    print('\nChosen model: ', chosen_model, sep='\n')

    # test chosen model
    model_name = next(iter(chosen_model))
    log_dir_model = os.path.join(args.out_log_dir, model_name)
    ckpt_filename = Project.get_ckpt_from_log_dir_model(log_dir_model)
    _, params = Project.load_ckpt_params(ckpt_filename, flatten=True)

    # TODO: Check if activations are in fact already sparsified.
    params['mode'] = 'test'
    params['ckpt_filename'] = ckpt_filename

    sys.stdout = open(os.devnull, "w")
    main_prog(copy.deepcopy(params))
    sys.stdout = sys.__stdout__

    print(f'\n=> Finished testing chosen model {params["model_name"]}.\n\n')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Sparsify ckpt model using an "optimal" slope '
        'threshold (highest threshold for which the train '
        'accuracy drop is within a specification). ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ckpt_filename',
                        metavar='ckpt_filename [STR]',
                        type=str,
                        help='')
    parser.add_argument('out_log_dir',
                        metavar='out_log_dir [STR]',
                        type=str,
                        help='Output log directory for sparsified model.')
    parser.add_argument('acc_drop_threshold',
                        metavar='acc_drop_threshold [FLOAT(-1, 0)]',
                        type=ArgCheck.n_float,
                        default=-0.25,
                        help='Maximum train accuracy percentage drop '
                        'allowed for sparsification. (default: %(default)s)')

    args = parser.parse_args()

    if not os.path.isdir(args.out_log_dir):
        os.makedirs(args.out_log_dir)

    if args.acc_drop_threshold <= -1:
        raise argparse.ArgumentTypeError(
            f'{args.acc_drop_threshold} should be in (-1, 0).')

    sparsify_with_optimal_knot_threshold(args)
