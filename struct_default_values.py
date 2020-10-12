default_values = {'mode' : 'train',
                  'net' : 'resnet32',
                  'model_name' : 'deepspline',
                  'device' : 'cuda:0',
                  'num_epochs' : 300,
                  'activation_type' : 'deepBspline_explicit_linear',
                  'spline_init' : 'leaky_relu',
                  'spline_size' : [51],
                  'spline_range' : 4,
                  'S_apl': 1,
                  'hidden' : 2,
                  'hyperparam_tuning' : False,
                  'lipschitz' : False,
                  'lmbda' : 1e-4,
                  'outer_norm' : 1,
                  'beta' : 0.001,
                  'optimizer' : ['SGD', 'Adam'],
                  'lr' : 1e-1,
                  'aux_lr' : 1e-3,
                  'weight_decay' : 5e-4,
                  'gamma' : 0.1,
                  'milestones' : [150, 225, 262],
                  'multires_milestones' : None,
                  'reset_multires' : False,
                  'resume' : False,
                  'resume_from_best' : False,
                  'ckpt_filename' : None,
                  'ckpt_nmax_files' : 3, # max number of saved *_net_*.ckpt
                                         # checkpoint files at a time. Set to -1 if not restricted. '
                  'log_dir' : './ckpt',
                  'log_step' : 50,
                  'valid_log_step' : 352,
                  'sparsify_activations' : False,
                  'slope_threshold' : 0.,
                  'seed' : (-1),
                  'test_as_valid' : False,
                  'dataset_name' : 'cifar10',
                  'data_dir' : './data',
                  'batch_size' : 128,
                  'num_workers' : 4, # Number of subprocesses to use for data loading.
                  'plot_imgs' : False,
                  'save_imgs' : False,
                  'save_title' : None,
                  'tensorboard' : False,
                  'verbose' : False,
                  'combination_str' : None,
                  'additional_info' : [],
                  'initial_spline_size' : None,
                  'num_classes' : None}


# This tree defines the strcuture of self.params in the Project() class.
# if it is desired to keep an entry in the first level that is also a leaf of deeper
# levels of the structure, this entry should be added to the first level too
# (as done for 'dataloader')
structure = {   'log_dir' : None,
                'model_name' : None,
                'verbose' : None,
                'net' : None,
                'dataset':
                    {
                    'dataset_name' : None,
                    'log_dir' : None,
                    'model_name' : None,
                    'plot_imgs' : None,
                    'save_imgs' : None,
                    'save_title' : None,
                    },
                'dataloader':
                    {
                    'data_dir' : None,
                    'batch_size' : None,
                    'num_workers' : None,
                    'train' :
                        {'seed' : None,
                         'test_as_valid' : None},
                    },
                'model':
                    {
                    'dataset_name' : None,
                    'activation_type' : None,
                    'spline_init' : None,
                    'initial_spline_size' : None,
                    'spline_size' : None,
                    'spline_range' : None,
                    'S_apl' : None,
                    'hidden' : None,
                    'slope_threshold' : None,
                    'hyperparam_tuning' : None,
                    'lipschitz' : None,
                    'weight_decay' : None,
                    'lmbda' : None,
                    'outer_norm' : None,
                    'beta' : None,
                    'net' : None,
                    'verbose' : None
                    }
            }
