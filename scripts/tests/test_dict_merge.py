#!/usr/bin/env python3

from ds_utils import dict_recursive_merge

params = {  'lmbda' : 10,
            "dataloader":
                {'bs' : 2,
                'train': {'c' : 2}
                },
            "model":
                {"dropout": 0.3,
                'deepspline':
                    {'size' : 3, 'grid' : 0.01}
                },
            "contrast" : 3
        }

user_params = { "dataloader": {'bs': 3},
                "model":
                {
                "dropout" : 0.4,
                'deepspline': {'size' : 1}
                },
                "contrast" : 4
            }

expected_merge_params = {'lmbda' : 10,
                        "dataloader":
                            {'bs' : 3,
                            'train': {'c' : 2}
                            },
                        "model":
                            {"dropout" : 0.4,
                            'deepspline':
                                {'size' : 1, 'grid' : 0.01}
                            },
                        "contrast" : 4
                        }


params_out = dict_recursive_merge(params, user_params)
print('Expected result : \n', expected_merge_params)
print('Result : \n', params_out)
