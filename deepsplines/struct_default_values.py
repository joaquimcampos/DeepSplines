default_values = {
    'mode': 'train',
    'net': 'resnet32_cifar',
    'device': 'cuda:0',
    'num_epochs': 300,
    'activation_type': 'deepBspline_explicit_linear',
    'spline_init': 'leaky_relu',
    'spline_size': 51,
    'spline_range': 4,
    'save_memory': False,
    'knot_threshold': 0.,
    'num_hidden_layers': 2,
    'num_hidden_neurons': 4,
    'lipschitz': False,
    'lmbda': 1e-4,
    'optimizer': ['SGD', 'Adam'],
    'lr': 1e-1,
    'aux_lr': 1e-3,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'milestones': [150, 225, 262],
    'log_dir': './ckpt',
    'model_name': 'deepspline',
    'resume': False,
    'resume_from_best': False,
    'ckpt_filename': None,
    'ckpt_nmax_files': 3,  # max number of saved *_net_*.ckpt
    # checkpoint files at a time. Set to -1 if not restricted. '
    'log_step': None,
    'valid_log_step': None,
    'seed': (-1),
    'test_as_valid': False,
    'dataset_name': 'cifar10',
    'data_dir': './data',
    'batch_size': 128,
    # Number of subprocesses to use for data loading.
    'num_workers': 4,
    'plot_imgs': False,
    'save_imgs': False,
    'verbose': False,
    'additional_info': [],
    'num_classes': None
}

# This tree defines the strcuture of self.params in the Project() class.
# if it is desired to keep an entry in the first level that is also a
# leaf of deeper levels of the structure, this entry should be added to
# the first level too (as done for 'dataloader')
structure = {
    'log_dir': None,
    'model_name': None,
    'verbose': None,
    'net': None,
    'knot_threshold': None,
    'dataset': {
        'dataset_name': None,
        'log_dir': None,
        'model_name': None,
        'plot_imgs': None,
        'save_imgs': None,
    },
    'dataloader': {
        'data_dir': None,
        'batch_size': None,
        'num_workers': None,
        'test_as_valid': None,
        'seed': None
    },
    'model': {
        'activation_type': None,
        'spline_init': None,
        'spline_size': None,
        'spline_range': None,
        'save_memory': None,
        'knot_threshold': None,
        'num_hidden_layers': None,
        'num_hidden_neurons': None,
        'net': None,
        'verbose': None
    }
}
