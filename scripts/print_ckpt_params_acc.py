#!/usr/bin/env python3
'''
This script prints the parameters and validation accuracy of a model.
The model is fetched from a checkpoint file (.pth) given as input.
If params['knot_threshold'] > 0., it also prints:
- knot threshold
- sparsity (int)
- lipschitz_bound
'''

import torch
import argparse

from deepsplines.project import Project
from deepsplines.manager import Manager


def print_ckpt_params_acc(args):
    """
    Args:
        args: verified arguments from arparser
    """
    ckpt, params = Project.load_ckpt_params(args.ckpt_filename)

    print('\nLoading checkpoint :', args.ckpt_filename, sep='\n')
    print('\nParameters : ', params, sep='\n')

    print('ckpt validation accuracy : {:.3f}%'.format(ckpt['valid_acc']))

    if params['knot_threshold'] > 0.:
        device = params['device']
        if device.startswith('cuda') and not torch.cuda.is_available():
            # TODO: Test how to load model on cpu trained on gpu
            raise OSError('cuda not available...')

        net = Manager.build_model(params, device=device)
        net.load_state_dict(ckpt['model_state'], strict=True)
        net.eval()

        print('knot_threshold : {:.4f}'.format(params['knot_threshold']))

        sparsity = net.compute_sparsity()
        print('sparsity : {:d}'.format(sparsity))

        lipschitz_bound = net.lipschitz_bound()
        print('lipschitz_bound : {:.3f}'.format(lipschitz_bound))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ckpt_filename',
                        metavar='ckpt_filename [STR]',
                        type=str,
                        help='')
    args = parser.parse_args()

    print_ckpt_params_acc(args)
