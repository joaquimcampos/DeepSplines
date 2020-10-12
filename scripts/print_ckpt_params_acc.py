#!/usr/bin/env python3

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
    params = ckpt['params']
    acc = ckpt['valid_acc'] if 'valid_acc' in ckpt else ckpt['acc']
    best_acc = ckpt['best_valid_acc']

    print('\nLoading parameters from checkpoint : ', args.ckpt_filename, sep='\n')
    print('\nParameters : ', params, sep='\n')
    print('Accuracy : {:.3f}%'.format(acc))

    if 'sparsify_activations' in params and params['sparsify_activations'] is True:
        net  = Manager.build_model(params, 'cuda:0')
        net.load_state_dict(ckpt['model_state'], strict=True)
        net.eval()

        slope_threshold = net.slope_threshold
        sparsity = net.compute_sparsity()
        lipschitz_bound = net.lipschitz_bound()

        print('slope_threshold : {:.4f}'.format(slope_threshold))
        print('sparsity : {:d}'.format(sparsity))
        print('lipschitz_bound : {:.3f}'.format(lipschitz_bound))
