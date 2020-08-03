import argparse
import os
import json
from ds_utils import ArgCheck
from scripts.search_run import SearchRun


class TorchDatasetSearchRun(SearchRun):
    """ Helper class for abstracting common operations in deepspline
    and standard grid searches with cifar10/mnist dataset.
    """

    def __init__(self, args):
        """ """
        super().__init__(args)



    @staticmethod
    def add_default_args(parser, is_standard=False):
        """ Add default args

        Args:
            is_standard: True, if activation is standard.
        """
        parser = SearchRun.add_default_args(parser)

        dataset_choices = {'cifar10', 'mnist'}
        parser.add_argument('dataset', metavar=f'dataset[STR]', type=str,
                    choices=dataset_choices, help=f'{str(dataset_choices)}')
        net_choices = {'simplenet', 'resnet20', 'resnet32'}
        parser.add_argument('--net', metavar=f'STR', type=str, default='resnet20',
                            choices=net_choices,
                            help=f'Network (for cifar10). Choices: {str(net_choices)}')
        parser.add_argument('--lr', metavar='FLOAT,>0', type=ArgCheck.p_float,
                            help=f'lr for main optimizer (network parameters).')

        if is_standard is False:
            parser.add_argument('--spline_size', metavar='LIST[INT>0]', nargs='+',
                                type=ArgCheck.p_odd_int, default=[33],
                                help='Number of activation coefficients.')
            parser.add_argument('--spline_range', metavar='FLOAT,>0', default=3,
                                type=ArgCheck.p_float, help='one-sided deepspline range.')
            optim_choices = {'SGD', 'mixed', 'SGD_SGD'}
            parser.add_argument('--optimizer', metavar='STR', type=str, default='mixed',
                                choices=optim_choices,
                                help=f'{str(optim_choices)} (for resnet only).')
            parser.add_argument('--aux_lr', metavar='FLOAT,>0', type=ArgCheck.p_float,
                                help=f'lr for aux optimizer (deepspline parameters).')
            parser.add_argument('--multires_milestones', metavar='LIST[INT,>0]',
                                nargs='+', type=ArgCheck.p_int,
                                help='Milestones for spline multi-resolution scheduling.')
            parser.add_argument('--lipschitz', action='store_true',
                                help='Perform lipschitz BV(2) regularization.')
            parser.add_argument('--hyperparam_tuning', action='store_true',
                                help='Tune hyperparameters.')
            parser.add_argument('--outer_norm', metavar='INT,>0', type=ArgCheck.p_int,
                                choices=[1,2], default=1, help='Outer tv/bv norm. Choices: {1,2}.')

        return parser



    def default_params(self, activation_type):
        """ Return default params, common to deepspline and standard gridsearch.
        """
        assert activation_type in ['deepBspline', 'deepRelu', 'deepBspline_explicit_linear', \
                                'hybrid_deepspline', 'deepBspline_superposition', \
                                'relu', 'leaky_relu', 'prelu']

        if self.args.dataset == 'cifar10':
            net = self.args.net
            batch_size = 128
            log_step, valid_log_step = 50, 352
            milestones = [150, 225, 262]
            num_epochs = 300
        else:
            # mnist
            net = 'simplestnet'
            batch_size = 64
            log_step, valid_log_step = 100, 844
            milestones = [22, 31, 36]
            num_epochs = 40

        # twoDnet training parameters
        params = {'net': net, 'device': 'cuda:0', 'activation_type' : activation_type,
                'num_epochs': num_epochs, 'milestones' : milestones,
                'log_step': log_step, 'valid_log_step': valid_log_step,
                'seed': 15,
                'dataset_name': self.args.dataset, 'batch_size': batch_size,
                'num_workers': 4, 'tensorboard': True}

        if 'deep' in activation_type:
            params['multires_milestones'] = self.args.multires_milestones
            params['spline_init'] = 'leaky_relu' # TODO: Allow change
            params['spline_size'] = self.args.spline_size
            params['spline_range'] = self.args.spline_range
            if activation_type == 'deepBspline_explicit_linear':
                params['multires_milestones'] = self.args.multires_milestones

            params['hyperparam_tuning'] = self.args.hyperparam_tuning
            params['lipschitz'] = self.args.lipschitz
            params['weight_decay'] = 5e-4
            params['outer_norm'] = self.args.outer_norm
        else:
            params['lmbda'] = 0


        if net in ['simplenet', 'simplestnet']:

            default_lr = 1e-2 if net == 'simplestnet' else 1e-3
            params['optimizer'] = ['Adam']
            params['lr'] = default_lr
            if self.args.lr is not None:
                params['lr'] = self.args.lr

            if 'deep' in activation_type:
                params['optimizer'] = ['Adam', 'Adam']
                params['aux_lr'] = default_lr
                if self.args.aux_lr is not None:
                    params['aux_lr'] = self.args.aux_lr
        else:
            params['optimizer'] = ['SGD']
            params['lr'] = 1e-1
            if self.args.lr is not None:
                params['lr'] = self.args.lr

            if 'deep' in activation_type:
                if self.args.optimizer == 'mixed':
                    # SGD optimizer for network parameters, Adam for deepspline parameters
                    params['optimizer'] = ['SGD', 'Adam']
                elif self.args.optimizer == 'SGD_SGD':
                    params['optimizer'] = ['SGD', 'SGD']

                params['aux_lr'] = 1e-3
                if self.args.aux_lr is not None:
                    params['aux_lr'] = self.args.aux_lr


        super_params = super().default_params()
        params = {**params, **super_params}

        return params
