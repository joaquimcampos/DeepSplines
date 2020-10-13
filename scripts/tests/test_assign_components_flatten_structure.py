#!/usr/bin/env python3

from ds_utils import assign_structure_recursive, flatten_structure

structure = {   'spline_range' : None,
                'dataset': {'dataset_name' : None},
                'dataloader':
                    {'spline_range' : None,
                    'train' : {'seed': None}
                    },
                'model':
                    {'deepspline':
                        {'spline_size': None,
                        'spline_range': None
                        }
                    }
            }


params = {'mode': 'train', 'net': None, 'model_name': None, 'num_epochs': 301,
        'spline_size': 121, 'spline_range': 3, 'log_step': None,
        'seed': 0, 'dataset_name': None, 'data_dir': None, 'batch_size': None}


# Test assign_structure_recursive
params_out = assign_structure_recursive(params, structure)
assert params_out['spline_range'] == params['spline_range']
assert params_out['dataloader']['spline_range'] == params['spline_range']
assert params_out['model']['deepspline']['spline_range'] == params['spline_range']

assert params_out['dataset']['dataset_name'] == params['dataset_name']
assert 'dataset_name' not in params_out

assert params_out['dataloader']['train']['seed'] == params['seed']
assert 'seed' not in params_out

assert params_out['model']['deepspline']['spline_size'] == params['spline_size']
assert 'spline_size' not in params_out

assert 'dataset_name' not in params_out.keys()
assert 'seed' not in params_out.keys()
assert 'spline_size' not in params_out.keys()

assert len(params_out['dataset'].keys()) == 1
assert len(params_out['dataloader'].keys()) == 2
assert len(params_out['dataloader']['train'].keys()) == 1
assert len(params_out['model'].keys()) == 1
assert len(params_out['model']['deepspline'].keys()) == 2


# Test flatten_structure
params_out_flattened = flatten_structure(params_out)
assert len(params_out_flattened) == len(params)

x, y = params_out_flattened, params
shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
assert len(shared_items) == len(x)
