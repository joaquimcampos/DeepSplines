#!/usr/bin/env python3

import torch
import numpy as np
from models.basemodel import MultiResScheduler
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear

args_dict = {'mode': 'linear', 'num_activations': 2,
            'size': 5, 'grid': 0.1,
            'init': 'leaky_relu', 'device': 'cpu'}

activ = DeepBSplineExplicitLinear(**args_dict)
multires_milestones, order = [10, 20, 30], 2
multires_scheduler = MultiResScheduler(multires_milestones, order)

step, size, grid = 1., 5, 0.1
coeffs_get = lambda x: torch.cat((torch.arange(-2, 2 + x, x).float(),
                                        torch.arange(-4, 4 + 2*x, 2*x).float()))
activ.coefficients_vect.data = coeffs_get(step)

for epoch in range(40):
    prev_coefficients_vect = activ.coefficients_vect.cpu().clone()
    multires_scheduler.step(epoch, activ)

    if epoch in multires_milestones:
        step, grid = step/2, grid/2
        size = order * (size-1) + 1
        new_coefficients_vect = coeffs_get(step)

        assert activ.size == size
        assert np.allclose(grid, activ.grid[0].item())
        assert torch.allclose(activ.coefficients_vect.data,
                            new_coefficients_vect, atol=1e-7), f'epoch {epoch}'
    else:
        assert torch.allclose(activ.coefficients_vect.data,
                            prev_coefficients_vect, atol=1e-7), f'epoch {epoch}'
