#!/usr/bin/env python3

"""
Main module that parses and verifies input arguments and
sends them to Manager for training/testing.
"""

import os
import argparse
from manager import Manager
from project import Project

from ds_utils import ArgCheck, assign_tree_structure
from struct_default_values import structure, default_values


# TODO: write setup.py!
# TODO: DO not push data. remove dependency on number of points.
def get_arg_parser():
    """
    Parses command-line arguments.

    The default values are fetched from the 'default_values' dictionary.
    (see struct_default_values.py)

    Returns:
        parser (argparse.ArgumentParser)
    """
    parser = argparse.ArgumentParser(description='Deep Spline Neural Network')

    # for test mode, need to provide --ckpt_filename
    parser.add_argument('--mode', choices=['train', 'test'], type=str,
                        help=f'Train or test mode. Default: {default_values["mode"]}.')

    # add other networks here, in the models/ directory and in Manager.build_model()
    net_choices = {'twoDnet_onehidden', 'twoDnet_onehidden', \
                    'resnet32_cifar', 'nin_cifar', 'convnet_mnist'}
    parser.add_argument('--net', metavar='STR', type=str,
                        help=f'Network to train. Available networks: {str(net_choices)}.'
                            'Default: {default_values["net"]}.')
    parser.add_argument('--model_name', metavar='STR', type=str,
                        help=f'Default: {default_values["model_name"]}.')
    parser.add_argument('--device', choices=['cuda:0', 'cpu'], type=str,
                        help=f'Default: {default_values["device"]}.')

    parser.add_argument('--num_epochs', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of epochs. Default: {default_values["num_epochs"]}.')

    # model parameters
    activation_type_choices = {'deepBspline', 'deepReLUspline', 'deepBspline_explicit_linear', \
                                'relu', 'leaky_relu'}
    parser.add_argument('--activation_type', choices=activation_type_choices, type=str,
                        help=f'Default: {default_values["activation_type"]}.')

    spline_init_choices = {'leaky_relu', 'relu', 'even_odd'}
    parser.add_argument('--spline_init', choices=spline_init_choices, type=str,
                        help='Initialize the b-spline coefficients according to this function. '
                            f'Default: {default_values["spline_init"]}.')
    parser.add_argument('--spline_size', metavar='INT>0', type=ArgCheck.p_odd_int,
                        help='Number of spline coefficients. Default: {default_values["spline_size"]}.')
    parser.add_argument('--spline_range', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Range of spline representation. Default: {default_values["spline_range"]}.')

    # see deepBspline_base.py docstring for details on --save_memory tradeoff.
    parser.add_argument('--save_memory', action='store_true',
                        help='Use a memory-efficient deepsplines version (for deepBsplines only) '
                            'at the expense of additional running time.')

    parser.add_argument('--knot_threshold', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help='If nonzero, sparsify activations by eliminating knots whose slope '
                            f'change is below this value. Default: {default_values["knot_threshold"]}.')

    # neural net
    parser.add_argument('--hidden', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of hidden neurons in each layer (for twoDnet). Default: {default_values["hidden"]}.')

    # regularization
    parser.add_argument('--lipschitz', action='store_true',
                        help='Perform lipschitz BV(2) regularization; the hyperparameter is set by --lmbda.')
    parser.add_argument('--lmbda', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help=f'TV/BV(2) hyperparameter. Default: {default_values["lmbda"]}.')

    # optimizer
    optimizer_choices={'Adam', 'SGD'}
    parser.add_argument('--optimizer', metavar='LIST[STR]', nargs='+', type=str,
                        help='Can be one or two args. In the latter case, the first arg is the main '
                            'optimizer (for network parameters) and the second arg is the aux optimizer '
                            '(for deepspline parameters). An "aux" optimizer different from "SGD" is usually required '
                            'for stability during training (Adam is recommended). Choices: {str(optimizer_choices)}.'
                            f'Default: {default_values["optimizer"]}.')

    parser.add_argument('--lr', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Learning rate for main optimizer (for network parameters). Default: {default_values["lr"]}.')
    parser.add_argument('--aux_lr', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Learning rate for aux optimizer (for deepspline parameters). Default: {default_values["aux_lr"]}.')
    parser.add_argument('--weight_decay', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help=f'weight decay hyperparameter. Default: {default_values["weight_decay"]}.')

    # multistep scheduler
    parser.add_argument('--gamma', metavar='FLOAT,[0,1]', type=ArgCheck.frac_float,
                        help=f'Learning rate decay. Default: {default_values["gamma"]}.')
    parser.add_argument('--milestones', metavar='LIST[INT,>0]', nargs='+', type=ArgCheck.p_int,
                        help='Milestones for multi-step lr_scheduler. Set to a single value higher '
                            'than num_epochs to not lower the learning rate during training. '
                            f'Default: {default_values["milestones"]}.')

    # logs-related
    parser.add_argument('--ckpt_filename', metavar='STR', type=str,
                    help=f'Continue training (if --mode=train) or test (if --mode=test) the model saved '
                        f'in this checkpoint. Default: {default_values["ckpt_filename"]}.')

    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume training from latest checkpoint. Need to provide '
                            '--model_name and --log_dir where the model is saved.')
    parser.add_argument('--resume_from_best', '-best', action='store_true',
                        help='Resume training from best validation accuracy checkpoint. Need to provide '
                            '--model_name and --log_dir where the model is saved.')

    parser.add_argument('--log_dir', metavar='STR', type=str,
                        help=f'Directory for saving checkpoints. Default: {default_values["log_dir"]}.')

    parser.add_argument('--log_step', metavar='INT,>0', type=ArgCheck.p_int,
                            help='Train log step in number of batches. '
                            'If None, done at every epoch. '
                            f'Default: {default_values["log_step"]}.')
    parser.add_argument('--valid_log_step', metavar='INT', type=int,
                        help='Validation log step in number of batches. '
                            'If None, done halfway and at the end of training. '
                            'If negative, done at every epoch. '
                            f'Default: {default_values["valid_log_step"]}.')

    # dataloader
    parser.add_argument('--seed', metavar='INT,>0', type=ArgCheck.nn_int,
                        help=f'fix seed for reproducibility. Default: {default_values["seed"]}.')
    parser.add_argument('--test_as_valid', action='store_true',
                        help='Train on full training data and evaluate model on test set in validation step.')

    # add other datasets here and create a corresponding Dataset class in datasets.py
    dataset_choices = {'cifar10', 'cifar100', 'mnist', 's_shape_1500', 'circle_1500'}
    parser.add_argument('--dataset_name', metavar='STR', type=str,
                        help=f'dataset to train/test on. Available datasets: {str(dataset_choices)}. '
                            f'Default: {default_values["dataset_name"]}.')

    parser.add_argument('--data_dir', metavar='STR', type=str, help=f'location of the data. Default: {default_values["data_dir"]}.')
    parser.add_argument('--batch_size', metavar='INT,>0', type=ArgCheck.p_int, help=f'Default: {default_values["batch_size"]}.')

    # dataset
    parser.add_argument('--plot_imgs', action='store_true', help='Plot train/test images')
    parser.add_argument('--save_imgs', action='store_true', help='Save train/test images')

    parser.add_argument('--verbose', '-v', action='store_true', help='Print more info.')

    additional_info_choices = {'sparsity', 'lipschitz_bound'}
    parser.add_argument('--additional_info', metavar='LIST[STR]', nargs='+', type=str,
                        help=f'Additional info to log in results json file. '
                            f'Choices: {str(additional_info_choices)}. Default: {default_values["additional_info"]}.')

    return parser



def verify_params(params):
    """
    Verifies the parameters (e.g. checks for mutual inclusivity and exclusivity).

    If not specified by the user via the command-line, a parameter
    gets the default value from the 'default_values' dictionary.
    (see struct_default_values.py).

    It also returns a dictionary with the parameters that were specified
    by the user, which are used to override the parameters saved in a
    checkpoint when resuming training or testing a model.

    Args:
        params (dict):
            dictionary with parameter names (keys) and values.

    Returns:
        params (dict):
            dictionary with all parameters. If not specified by the user,
            a parameter gets the default value in the 'default_values'
            dictionary.
        user_params (dict):
            dictionary with the user-defined parameters (command-line args).
    """
    user_params = {}  # parameters input by user

    # Check parameters input by user and set default values
    for key, value in default_values.items():
        if key not in params or params[key] is None:
            params[key] = value # param which was not set by user
        elif params[key] is not False:
            user_params[key] = params[key] # param or action='store_true' flag input by user

    # check parameter dependecies
    if params['mode'] == 'test' and 'ckpt_filename' not in user_params:
        if 'log_dir' in user_params and 'model_name' in user_params:
            # test on last model checkpoint
            log_dir_model = os.path.join(params['log_dir'], params['model_name'])
            params['ckpt_filename'] = Project.get_ckpt_from_log_dir_model(log_dir_model)
            user_params['ckpt_filename'] = params['ckpt_filename']
        else:
            raise ValueError('Please provide --ckpt_filename for testing.')

    if params['resume_from_best']:
        params['resume'] = True  # set 'resume' to True if 'resume_from_best' is True
        user_params['resume'] = True

    if len(params['optimizer']) > 2:
        raise ValueError('Please provide a maximum of two optimizers (main and aux).')

    if params['resume'] and ('log_dir' not in user_params or 'model_name' not in user_params):
        raise ValueError('Need to provide either log_dir and model_name, '
                        'if resuming training from best or latest checkpoint.')

    if params['activation_type'] == 'deepReLUspline' and params['spline_init'] == 'even_odd':
        raise ValueError('Cannot use even_odd spline initialization with deepReLUspline.')

    if params['save_memory'] is True and not params['activation_type'].startswith('deepBspline'):
        raise ValueError('--save_memory can only be set when using deepBsplines.')

    if params['knot_threshold'] > 0. and 'deep' not in params['activation_type']:
        raise ValueError('--knot_threshold can only be set when using deepsplines.')

    return params, user_params



def main_prog(params, isloaded_params=False):
    """
    Main program that initializes the Manager
    with the parameters and runs the training or testing.

    It first verifies the params dictionary (all parameters) and the
    'user_params' (user-defined parameters), if necessary.
    'user_params' is used to override the parameters saved in a
    checkpoint, if resuming training or testing a model.

    'params' and 'user_params' are then assigned a tree structure
    according to the 'structure'
    dictionary (see struct_default_values.py).

    Finally, 'params' and 'user_params' are used to instantiate
    a Manager() object and the training or testing is ran.

    Args:
        params (dict):
            dictionary with the parameters from the parser.
        isloaded_params :
            True if params were loaded from ckpt (no need to verify) and
            are flattened (ds_utils), i.e., don't have a tree structure.
    """
    if isloaded_params:
        assert params['mode'] != 'test', 'params need to be verified in test mode...'
        user_params = params
    else:
        params, user_params = verify_params(params)

    # assign tree structure to params according to 'structure' dictionary
    # (see struct_default_values.py)
    user_params = assign_tree_structure(user_params, structure)
    params = assign_tree_structure(params, structure)

    manager = Manager(params, user_params)

    if params['mode'] == 'train':
        print('\n==> Training model.')
        manager.train()

    elif params['mode'] == 'test':
        print('\n\n==> Testing model.')
        manager.test()



if __name__ == "__main__":

    # parse arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args) # transform to dictionary

    main_prog(params)
