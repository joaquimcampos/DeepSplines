import argparse
import os
import json
from ds_utils import ArgCheck
from scripts.search_run import SearchRun


class TwoDSearchRun(SearchRun):
    """ Helper class for abstracting common operations in deepspline
    and standard grid searches with twoD dataset.
    """

    def __init__(self, args):
        """ """
        super().__init__(args)



    @staticmethod
    def add_default_args(parser, is_deepspline=False, is_apl=False):
        """ Add default args

        Args:
            is_deepspline: True, if activation is standard.
        """
        parser = SearchRun.add_default_args(parser)

        dataset_choices = {'s_shape_1500', 'circle_1500'}
        parser.add_argument('dataset', type=str, metavar='dataset[STR]',
                            choices=dataset_choices, help=f'{str(dataset_choices)}')
        net_choices = {'twoDnet_onehidden', 'twoDnet_twohidden'}
        parser.add_argument('net', metavar=f'net[STR]', type=str,
                            choices=net_choices, help=f'{str(net_choices)}')
        parser.add_argument('--device', type=str, choices=['cuda:0', 'cpu'],
                            default='cpu', help=' ')
        parser.add_argument('--hidden', metavar='INT,>0', type=ArgCheck.p_int,
                            default=4, help=f'Number of hidden neurons for the network.')
        num_epochs_choices = {500, 1000}
        parser.add_argument('--num_epochs', metavar='INT,>0', type=ArgCheck.p_int, default=500,
                            choices=num_epochs_choices, help=f'{str(num_epochs_choices)}')

        if is_apl is True:
            parser.add_argument('--S_apl', metavar='INT,>0', type=ArgCheck.p_int,
                            default=1, help=f'Additional number of APL knots.')

        if is_deepspline is True:
            parser.add_argument('--spline_size', metavar='INT>0',
                                type=ArgCheck.p_odd_int, default=21,
                                help='Number of activation coefficients.')
            parser.add_argument('--spline_range', metavar='FLOAT,>0', default=1,
                                type=ArgCheck.p_float, help='one-sided deepspline range.')
            parser.add_argument('--lipschitz', action='store_true',
                                help='Perform lipschitz BV(2) regularization.')
            parser.add_argument('--outer_norm', metavar='INT,>0', type=ArgCheck.p_int,
                                choices=[1,2], default=1, help='Outer tv/bv norm. Choices: {1,2}.')

        return parser



    def default_params(self, activation_type):
        """ Return default params, common to deepspline and standard gridsearch.
        """
        assert activation_type in ['deepBspline', 'deepBspline_explicit_linear', \
                                    'hybrid_deepspline', \
                                    'apl', 'relu', 'leaky_relu', 'prelu']

        milestones = [440, 480] if self.args.num_epochs == 500 else [830, 950]
        batch_size = 10
        num_train_samples = int(self.args.dataset.split('_')[-1])
        log_step = int(num_train_samples / batch_size) # at every epoch
        valid_log_step = int((log_step * self.args.num_epochs) / 2) # in the middle and at the end

        # twoDnet training parameters
        params = {'net': self.args.net, 'device': self.args.device,
                'activation_type': activation_type,
                'num_epochs': self.args.num_epochs, 'hidden' : self.args.hidden,
                'optimizer': ['Adam'], 'lr': 1e-3, 'milestones' : milestones,
                'log_step': log_step, 'valid_log_step': valid_log_step,
                'dataset_name': self.args.dataset, 'batch_size': batch_size,
                'save_imgs': True, 'num_workers': 2
                }

        if 'deep' in activation_type:
            params['spline_init'] = 'even_odd'
            params['spline_size'] = self.args.spline_size
            params['spline_range'] = self.args.spline_range
            params['hyperparam_tuning'] = True
            params['lipschitz'] = self.args.lipschitz
            params['weight_decay'] = 0 # tuned weight decay. Don't penalize non-weight parameters.
            params['outer_norm'] = self.args.outer_norm
        else:
            params['lmbda'] = 0

        if 'apl' in activation_type:
            params['S_apl'] = self.args.S_apl
        else:
            params['beta'] = 0


        super_params = super().default_params()
        params = {**params, **super_params}

        return params
