#!/usr/bin/env python3

import argparse
import os
import numpy as np
import sys
import copy
import shutil

from ds_utils import ArgCheck
from project import Project
from main import main_prog


def delete_model(log_dir, model_name):
    # delete model directory
    log_dir_model = os.path.join(log_dir, model_name)
    assert os.path.isdir(log_dir_model)
    shutil.rmtree(log_dir_model)
    # delete model entry in train_result.json
    del results_dict[model_name]
    Project.dump_results_dict(results_dict,log_dir)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Sparsify model runs using an "optimal" slope '
                                            'threshold (highest threshold for which the train '
                                            'accuracy drop is within a specification). '
                                            'Save resulting model sparsity/lipschitz_bound.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('log_dir', metavar='log_dir[STR]', type=str,
                        help='Log directory with model runs.')
    parser.add_argument('out_log_dir', metavar='out_log_dir[STR]', type=str,
                        help='Output log directory for sparsified models.')
    parser.add_argument('acc_drop_threshold', metavar='FLOAT,<0', type=ArgCheck.n_float, default=-0.25,
                        help='Maximum train accuracy percentage drop allowed for sparsification.')
    args = parser.parse_args()

    if not os.path.isdir(args.out_log_dir):
        os.makedirs(args.out_log_dir)


    results_dict = Project.load_results_dict(args.log_dir)

    for model in results_dict:

        print(f'\n=> Compute optimal threshold/sparsity for model run: {model}.')
        log_dir_model = os.path.join(args.log_dir, model)
        ckpt_filename = Project.get_ckpt_from_log_dir_model(log_dir_model)
        ckpt, params = Project.load_ckpt_params(ckpt_filename, flatten=True)

        # take the trained model, do one more epoch with a slope threshold
        # applied and the model frozen, and compute the train accuracy.
        params['resume'] = True
        params['num_epochs'] = ckpt['num_epochs_finished'] + 1
        params['valid_log_step'] = params['log_step'] # at each epoch
        params['log_dir'] = args.out_log_dir
        params['ckpt_filename'] = ckpt_filename

        params['additional_info'] = ['sparsity']
        if params['lipschitz'] is True:
            params['additional_info'].append('lipschitz_bound')

        base_model_name = params['model_name']
        params['sparsify_activations'] = True

        # variable initialization
        base_train_acc = 0.
        prev_model_name = None # previous model name
        chosen_model, chosen_threshold = None, None

        acc_drop_threshold = args.acc_drop_threshold # training accuracy maximum allowed percentage drop
        threshold_list = np.concatenate((np.zeros(1),
                                        np.arange(0.0002, 0.004, 0.0002),
                                        np.arange(0.004, 1, 0.05),
                                        np.arange(1, 3, 0.2),
                                        np.arange(3, 10, 0.5),
                                        np.arange(10, 100, 2)))

        for k in range(threshold_list.shape[0]):

            threshold = threshold_list[k]
            params['model_name'] = base_model_name + '_slope_diff_threshold_{:.4f}'.format(threshold)
            params['slope_diff_threshold'] = threshold

            sys.stdout = open(os.devnull, "w")
            main_prog(copy.deepcopy(params), isloaded_params=True)
            sys.stdout = sys.__stdout__

            results_dict = Project.load_results_dict(args.out_log_dir)
            model_dict = results_dict[params['model_name']]

            if k == 0:
                assert np.allclose(threshold, 0)
                base_train_acc = model_dict['latest_train_acc']

            acc_drop = np.clip((model_dict['latest_train_acc'] - base_train_acc),
                                a_max=100, a_min=-100)

            print('\nThreshold: {:.4f}'.format(threshold))
            print('Accuracy drop: {:.3f}%'.format(acc_drop))
            print('Sparsity: {:d}'.format(int(model_dict['sparsity'])))

            if acc_drop < acc_drop_threshold or model_dict['sparsity'] == 0:
                # delete current model; chosen_model is the previous one
                delete_model(args.out_log_dir, params['model_name'])
                break
            else:
                if k > 0:
                    # delete previous model
                    delete_model(args.out_log_dir, prev_model_name)

                chosen_model = {params['model_name']: model_dict}
                chosen_threshold = params['slope_diff_threshold']
                prev_model_name = params['model_name']


        assert chosen_model is not None
        assert chosen_threshold is not None
        print('\nChosen model: ', chosen_model, sep='\n')

        # test model
        params['model_name'] = next(iter(chosen_model))
        params['slope_diff_threshold'] = chosen_threshold
        params['mode'] = 'test'
        sys.stdout = open(os.devnull, "w")
        main_prog(copy.deepcopy(params))
        sys.stdout = sys.__stdout__

        print(f'\n=> Finished testing chosen model {params["model_name"]}.\n\n')
