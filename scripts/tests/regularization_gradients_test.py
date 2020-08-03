#!/usr/bin/env python3

import argparse
import torch
from models.deepBspline import DeepBSpline
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear
from models.deepRelu import DeepReLU
from ds_utils import ArgCheck, spline_grid_from_range


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Test TV(2) gradients starting from relu.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', metavar='FLOAT,>0', type=ArgCheck.p_float, default=1e-6,
                        help=f'Learning rate for optimizer.')
    parser.add_argument('--activation_type', type=str,
                        choices={'deepBspline', 'deepRelu', 'deepBspline_explicit_linear'},
                        default='deepBspline', help=' ')
    parser.add_argument('--spline_size', metavar='INT>0',
                        type=ArgCheck.p_odd_int, default=51,
                        help='Number of b-spline/relu + linear coefficients.')
    args = parser.parse_args()


    size = args.spline_size
    spline_range = 1
    grid = spline_grid_from_range(size, spline_range)

    parab_range = 1
    args_dict = {'mode': 'linear', 'num_activations': 1,
                'size': size, 'grid': grid,
                'init': 'relu', 'device': 'cpu'}

    activation_type = args.activation_type

    if activation_type == 'deepBspline':
        activation = DeepBSpline(**args_dict)
    elif activation_type == 'deepBspline_explicit_linear':
        activation = DeepBSplineExplicitLinear(**args_dict)
    elif activation_type == 'deepRelu':
        activation = DeepReLU(**args_dict)

    lr = args.lr
    loss = activation.totalVariation().sum()
    loss.backward()
    print(f'\nInitial TV(2): \n{loss}')

    with torch.no_grad():
        if activation_type == 'deepRelu':
            activation.slopes -= lr * activation.slopes.grad
        else:
            activation.coefficients_vect -= lr * activation.coefficients_vect.grad

        new_loss = activation.totalVariation().sum()
        print(f'\nTV(2) after grad step with lr {lr}: \n{new_loss}')
