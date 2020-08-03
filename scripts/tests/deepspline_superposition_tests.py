#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from models.basemodel import BaseModel
from ds_utils import spline_grid_from_range


class TestModule(BaseModel):

    def __init__(self, **params):

        super().__init__(**params)
        self.fc1 = nn.Linear(1, 1)
        self.activation = self.init_activation(('linear', 1))
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        x = self.activation(self.fc1(x))
        return x


if __name__ == "__main__":

    args_dict = {'activation_type': 'deepBspline_superposition',
                'spline_size': [3, 5, 17], 'spline_range': 2,
                'spline_init': 'leaky_relu', 'device': 'cpu',
                'lipschitz': True, 'hyperparam_tuning': True,
                'lmbda': 1e-5, 'verbose': True, 'outer_norm':1}

    model = TestModule(**args_dict)
    assert model.activation.idx_steps == [8, 4, 1], f'{model.activation.idx_steps}'

    model.init_hyperparams()

    print('Deepspline parameters :')
    for name, param in model.named_parameters_deepspline():
        print(name, param)

    print('\nRemaining parameters :')
    for name, param in model.named_parameters_no_deepspline():
        print(name, param)

    names = ['spline_bias', 'spline_weight', 'coefficients_vect']
    assert all(name in names for name in model.activation.parameter_names())
    assert len(list(model.activation.parameter_names())) == 3

    model.activation.spline_superposition[1].coefficients_vect.data = \
                                    torch.tensor([2, 1, 2, 1, 0]).float()

    _, tv_bv_unweighted = model.TV_BV()
    assert np.allclose(tv_bv_unweighted.item(), 7.01), \
                f'tv_bv = {tv_bv_unweighted.item()} != 7.01'

    model.activation.spline_superposition[1].coefficients_vect.data = \
                                torch.tensor([-1, -2, -1, -2, -3]).float()

    _, tv_bv_unweighted = model.TV_BV()
    assert np.allclose(tv_bv_unweighted.item(), 5.01), \
            f'tv_bv = {tv_bv_unweighted.item()} != 5.01'

    threshold_sparsity, threshold_sparsity_mask = \
            model.activation.get_threshold_sparsity(threshold=1e-3)

    assert threshold_sparsity.sum() == 2
    expected_mask = torch.zeros(model.activation.size[-1] - 2).bool()
    expected_mask[torch.tensor([3, 7])] = True
    assert torch.allclose(threshold_sparsity_mask.long(), expected_mask.long())
