#!/usr/bin/env python3

import torch
from models.deepBspline import DeepBSpline
from models.deepRelu import DeepReLU


args_dict = {'mode': 'linear', 'num_activations': 2,
            'size': 5, 'grid': 0.1,
            'init': 'leaky_relu', 'device': 'cpu'}

deepBspline = DeepBSpline(**args_dict)
deepRelu = DeepReLU(**args_dict)

activation_list = [deepBspline, deepRelu]

for activ in activation_list:
    print(f'==>{activ.__class__.__name__}:')
    if activ.__class__.__name__ == 'DeepReLU':
        print(f'{activ.slopes}\n{activ.linear_coefficients}')
    else:
        print(f'{activ.coefficients_vect}')
    print('\n')

input = torch.tensor([[-0.15, 0.],
                    [0.1, -0.1],
                    [-0.025, 0.05]])

for activ in activation_list:
    x = input.clone().requires_grad_()
    y = activ(x)
    y.sum().backward()

    print(f'==>{activ.__class__.__name__}')
    if activ.__class__.__name__ == 'DeepReLU':
        print(f'slopes grad: \n{activ.slopes.grad}')
        print(f'linear_coefficients grad: \n{activ.linear_coefficients.grad}')
    else:
        print(f'coefficients_vect grad: \n{activ.coefficients_vect.grad}')

    print(f'input grad: \n{x.grad}')
    print(f'TV(2): {activ.totalVariation()}')
    print(f'fZerofOneAbs: {activ.fZerofOneAbs()}')
    print('\n')
