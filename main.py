#!/usr/bin/env python3

import os
import argparse
from manager import Manager
from project import Project

from ds_utils import ArgCheck, assign_structure_recursive
from struct_default_values import structure, default_values

# TODO: write setup.py!
def get_arg_parser():
    """ Define argument parser

    The default values are saved as a dictionary in struct_default_values.py
    """
    parser = argparse.ArgumentParser(description='Deep Spline Neural Network')

    # for test mode, need to provide --ckpt_filename
    parser.add_argument('--mode', choices=['train', 'test'], type=str,
                        help=f'Train or test mode. Default: {default_values["mode"]}.')

    # add other networks here, in the models/ directory and in Manager.build_model()
    net_choices = {'twoDnet_onehidden', 'twoDnet_onehidden', 'simplenet', 'simplestnet', \
                    'resnet20', 'resnet32', 'resnet32_linear', 'nin', 'nin_linear'}
    parser.add_argument('--net', metavar='STR', type=str,
                        help=f'Network to train. Available networks: {str(net_choices)}. Default: {default_values["net"]}.')
    parser.add_argument('--model_name', metavar='STR', type=str,
                        help=f'Default: {default_values["model_name"]}.')
    parser.add_argument('--device', choices=['cuda:0', 'cpu'], type=str,
                        help=f'Default: {default_values["device"]}.')

    parser.add_argument('--num_epochs', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of epochs. Default: {default_values["num_epochs"]}.')

    # model parameters
    activation_type_choices = {'deepBspline', 'deepRelu', 'deepBspline_explicit_linear', \
                                'apl', 'relu', 'leaky_relu', 'prelu'}
    parser.add_argument('--activation_type', choices=activation_type_choices, type=str,
                        help=f'Default: {default_values["activation_type"]}.')

    spline_init_choices = {'relu', 'leaky_relu', 'softplus', 'even_odd', 'random', 'identity', 'zero'}
    parser.add_argument('--spline_init', choices=spline_init_choices, type=str,
                        help='Initialize the b-spline coefficients according to this function. '
                            f'Default: {default_values["spline_init"]}.')
    parser.add_argument('--spline_size', metavar='INT>0', type=ArgCheck.p_odd_int,
                        help='Number of spline coefficients. Default: {default_values["spline_size"]}.')
    parser.add_argument('--spline_range', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Range of spline representation. Default: {default_values["spline_range"]}.')

    parser.add_argument('--S_apl', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Additional number of APL knots. Default: {default_values["S_apl"]}.')

    parser.add_argument('--hidden', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of hidden neurons in each layer (for twoDnet). Default: {default_values["hidden"]}.')

    # regularization
    parser.add_argument('--hyperparam_tuning', action='store_true',
                        help='Tune TV/BV(2) and weight decay hyperparameters according to '
                        'a tuning constant (--lmbda).')
    parser.add_argument('--lipschitz', action='store_true',
                        help='Perform lipschitz BV(2) regularization; the hyperparameter is set by --lmbda.')
    parser.add_argument('--lmbda', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help='if --hyperparam_tuning is set, it is used tune TV/BV(2) and weight decay '
                            'hyperparameters; otherwise, it is the TV/BV(2) hyperparameter.'
                            f'Default: {default_values["lmbda"]}.')
    parser.add_argument('--outer_norm', metavar='INT,>0', choices=[1,2], type=ArgCheck.p_int,
                        help='Outer norm for TV(2)/BV(2). Choices: {1,2}. '
                            f'Default: {default_values["outer_norm"]}.')

    parser.add_argument('--beta', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help='Weight decay on APL parameters. '
                            f'Default: {default_values["beta"]}.')

    # optimizer
    optimizer_choices={'Adam', 'SGD'}
    parser.add_argument('--optimizer', metavar='LIST[STR]', nargs='+', type=str,
                        help='Can be one or two args. In the latter case, the first arg is the main '
                            'optimizer (for network parameters) and the second arg is the aux optimizer '
                            '(for deepspline parameters). Adam aux_optimizer is usually required for stability '
                            f'during training, even if main optimizer is SGD. Choices: {str(optimizer_choices)}. '
                            f'Default: {default_values["optimizer"]}.')

    parser.add_argument('--lr', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Learning rate for main optimizer (for network parameters). Default: {default_values["lr"]}.')
    parser.add_argument('--aux_lr', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Learning rate for aux optimizer (for deepspline parameters). Default: {default_values["aux_lr"]}.')
    parser.add_argument('--weight_decay', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help=f'L2 penalty on parameters. If --hyperparam_tuning is set, '
                            'then a custom weight decay is applied to the weights of the network '
                            'and --weight_decay is applied to the remaining parameters (e.g. BatchNorm). '
                            'Default: {default_values["weight_decay"]}.')

    # multistep scheduler
    parser.add_argument('--gamma', metavar='FLOAT,[0,1]', type=ArgCheck.frac_float,
                        help=f'Learning rate decay. Default: {default_values["gamma"]}.')
    parser.add_argument('--milestones', metavar='LIST[INT,>0]', nargs='+', type=ArgCheck.p_int,
                        help='Milestones for multi-step lr_scheduler. Set to a single value higher than num_epochs '
                            'if you do not wish to lower the learning rate during training. '
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
                        help=f'Directory for saving checkpoints and tensorboard events. Default: {default_values["log_dir"]}.')

    parser.add_argument('--log_step', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Train log step in batch_size. Default: {default_values["log_step"]}.')
    parser.add_argument('--valid_log_step', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Validation step in epochs. Default: {default_values["valid_log_step"]}.')

    parser.add_argument('--sparsify_activations', action='store_true', help='Sparsify activations by eliminating slopes below threshold.')
    parser.add_argument('--slope_threshold', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help=f'Activation slopes threshold. Default: {default_values["slope_threshold"]}.')

    # dataloader
    parser.add_argument('--seed', metavar='INT,>0', type=ArgCheck.nn_int,
                        help=f'fix seed for reproducibility. Default: {default_values["seed"]}.')
    parser.add_argument('--test_as_valid', action='store_true',
                        help='Train on full training data and evaluate model on test set in validation step.')

    # add other datasets here and create a corresponding Dataset class in datasets.py
    dataset_choices = {'s_shape_1500', 'circle_1500', 'cifar10', 'cifar100', 'mnist'}
    parser.add_argument('--dataset_name', metavar='STR', type=str,
                        help=f'dataset to train/test on. Available datasets: {str(dataset_choices)}. '
                            f'Default: {default_values["dataset_name"]}.')

    parser.add_argument('--data_dir', metavar='STR', type=str, help=f'location of the data. Default: {default_values["data_dir"]}.')
    parser.add_argument('--batch_size', metavar='INT,>0', type=ArgCheck.p_int, help=f'Default: {default_values["batch_size"]}.')

    # dataset
    parser.add_argument('--plot_imgs', action='store_true', help='Plot train/test images')
    parser.add_argument('--save_imgs', action='store_true', help='Save train/test images')
    parser.add_argument('--save_title', metavar='STR', type=str,
                        help='Title for saving images. Predefined titles are used if not set.')

    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard logs.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print more info.')

    additional_info_choices = {'sparsity', 'lipschitz_bound'}
    parser.add_argument('--additional_info', metavar='LIST[STR]', nargs='+', type=str,
                        help=f'Additional info to log in results json file. '
                            f'Choices: {str(additional_info_choices)}. Default: {default_values["additional_info"]}.')

    return parser



def assign_recursive(params, user_params):
    """ Assign recursive structure """
    # assign recursive structure to both dictionaries as defined in struct_default_values.py
    user_params = assign_structure_recursive(user_params, structure)
    params = assign_structure_recursive(params, structure)

    return params, user_params



def verify_params(params):
    """ Verify parameters (e.g. mutual inclusivity or exclusivity) and
    assign recursive structure to parameters.
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

    return params, user_params



def main_prog(params, isloaded_params=False):
    """ Main program

    Args:
        isloaded_params : True if params are loaded from ckpt
                        (no need to verify) and flattened (ds_utils)
    """
    if isloaded_params:
        assert params['mode'] != 'test', 'params need to be verified in test mode...'
        user_params = params
    else:
        params, user_params = verify_params(params)

    # assign recursive structure to params according to structure in struct_default_values.py
    params, user_params = assign_recursive(params, user_params)

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
