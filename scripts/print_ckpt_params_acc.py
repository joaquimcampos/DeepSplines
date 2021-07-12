#!/usr/bin/env python3

'''
This script prints the parameters and validation accuracy
saved in the checkpoint (.pth) given as argument.
If sparsify_activations is True, it also prints:
- slope_diff_threshold
- sparsity (int)
- lipschitz_bound
'''

import argparse
from project import Project
from manager import Manager


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Load parameters from checkpoint file.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ckpt_filename', metavar='NAME', type=str, help='')
    args = parser.parse_args()

    ckpt = Project.get_loaded_ckpt(args.ckpt_filename)
    print('\nLoading parameters from checkpoint :', args.ckpt_filename, sep='\n')

    params = ckpt['params']
    print('\nParameters : ', params, sep='\n')

    print('ckpt validation accuracy : {:.3f}%'.format(ckpt['valid_acc']))

    if params['sparsify_activations'] is True:
        device = ckpt['params']['device']
        if device == 'cuda:0' and not torch.cuda.is_available():
            # TODO: Test how to load model on cpu trained on gpu
            raise OSError('cuda not available...')

        net  = Manager.build_model(ckpt['params'], device=device)
        net.load_state_dict(ckpt['model_state'], strict=True)
        net.eval()

        slope_diff_threshold = net.slope_diff_threshold
        sparsity = net.compute_sparsity()
        lipschitz_bound = net.lipschitz_bound()

        print('slope_diff_threshold : {:.4f}'.format(slope_diff_threshold))
        print('sparsity : {:d}'.format(sparsity))
        print('lipschitz_bound : {:.3f}'.format(lipschitz_bound))
