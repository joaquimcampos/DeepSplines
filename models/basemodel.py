import torch
from torch import nn
from torch import Tensor

from models.deepBspline import DeepBSpline
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear
from models.deepRelu import DeepReLU
from models.apl import APL
from ds_utils import spline_grid_from_range



class MultiResScheduler():
    """ Scheduler for multi-resolution approach
    for increasing spline coefficients.
    """

    def __init__(self, milestones, order=2):
        """
        Args:
            milestones: epochs for which to increase resolution.
            order: order of increase of the number of spline coefficients.
        """
        assert isinstance(milestones, list), 'milestones should be a list.'
        self.milestones = milestones
        self.gamma = order # for compatibility with scheduler terminology
        self.last_epoch = 0
        self.idx = 0


    def step(self, modules_deepspline, epoch=None):
        """ Increase resolution (number of coefficients) of
        deepspline modules.

        Args:
            modules_deepspline: iterator over deepspline modules.
        Returns:
            step_done: True if multires step was performed.
        """
        step_done = False
        if epoch is not None:
            self.last_epoch = epoch

        if self.idx < len(self.milestones) and self.last_epoch == self.milestones[self.idx]:
            for module in modules_deepspline:
                if isinstance(module, DeepBSplineExplicitLinear):
                    module.increase_resolution(order=self.gamma)
                else:
                    raise ValueError('Found deepspline module which is not DeepBSplineExplicitLinear...')

            self.idx += 1
            step_done = True

        self.last_epoch += 1

        return step_done



class BaseModel(nn.Module):

    def __init__(self, **params):
        """ """
        super().__init__()

        self.params = params

        self.set_attributes('activation_type', 'dataset_name',
                            'num_classes', 'device')
        # deepspline
        self.set_attributes('spline_init', 'spline_size',
                            'spline_range', 'S_apl', 'slope_threshold')
        # regularization
        self.set_attributes('hyperparam_tuning', 'outer_norm')

        self.spline_grid = spline_grid_from_range(self.spline_size,
                                                    self.spline_range)

        self.deepspline = None
        if self.activation_type == 'deepRelu':
            self.deepspline = DeepReLU
        elif self.activation_type == 'deepBspline':
            self.deepspline = DeepBSpline
        elif self.activation_type == 'deepBspline_explicit_linear':
            self.deepspline = DeepBSplineExplicitLinear

        self.apl = None
        if self.activation_type == 'apl':
            self.apl = APL


    def set_attributes(self, *names):
        """ """
        for name in names:
            assert isinstance(name, str), f'{name} is not string.'
            if name in self.params:
                setattr(self, name, self.params[name])


    ############################################################################
    # Activation initialization


    def init_activation_list(self, activation_specs, bias=True, **kwargs):
        """ Initialize list of activations

        Args:
            activation_specs : list of pairs ('layer_type', num_channels/neurons);
                                len(activation_specs) = number of activation layers;
                                e.g., [('conv', 64), ('linear', 100)].
            bias : explicit bias; only relevant for explicit_linear activations.
        """
        assert isinstance(activation_specs, list)

        if self.deepspline is not None:
            size, grid = self.spline_size, self.spline_grid
            activations = nn.ModuleList()
            for mode, num_activations in activation_specs:
                activations.append(self.deepspline(size=size, grid=grid, init=self.spline_init,
                                            bias=False, mode=mode, num_activations=num_activations, #bias=bias
                                            device=self.device))
        elif self.apl is not None:
            activations = nn.ModuleList()
            for mode, num_activations in activation_specs:
                activations.append(self.apl(mode=mode,
                                            num_activations=num_activations,
                                            S_apl=self.S_apl,
                                            device=self.device))
        else:
            activations = self.init_standard_activations(activation_specs)

        return activations



    def init_activation(self, activation_specs, **kwargs):
        """ Initialize a single activation

        Args:
            activation_specs: tuple, e.g., ('conv', 64)
        """
        assert isinstance(activation_specs, tuple)
        activation = self.init_activation_list([activation_specs], **kwargs)[0]

        return activation



    def init_standard_activations(self, activation_specs, **kwargs):
        """ Initialize non-spline activations and puts them in nn.ModuleList()

        Args:
            activation_type : 'relu', 'leaky_relu', 'prelu'.
            activation_specs : list of pairs ('layer_type', num_channels/neurons);
                                len(activation_specs) = number of activation layers.
        """
        activations = nn.ModuleList()

        if self.activation_type == 'relu':
            relu = nn.ReLU()
            for i in range(len(activation_specs)):
                activations.append(relu)

        elif self.activation_type == 'leaky_relu':
            leaky_relu = nn.LeakyReLU()
            for i in range(len(activation_specs)):
                activations.append(leaky_relu)

        elif self.activation_type == 'prelu':
            for _, num_channels in activation_specs:
                activations.append(nn.PReLU(num_parameters=num_channels))

        else:
            raise ValueError(f'{self.activation_type} is not in relu family...')

        return activations



    def initialization(self, control='backprop', init_type='He'):
        """ """
        assert control in ['forward', 'backprop'], ('Can either control variance of'
                                                    'forward pass or backprop')
        assert init_type in ['He', 'Xavier', 'custom_normal']

        if init_type == 'Xavier':
            print('\nUsing Xavier init...')
            nonlinearity = 'random' # init for random activation = Xavier init

        elif init_type == 'custom_normal':
            print('\nUsing custom Gauss(0, 0.05) weight initialization...')
            nonlinearity = 'custom_normal'

        elif self.activation_type == 'prelu':
            nonlinearity = 'leaky_relu'
            slope_init = 0.25 # default nn.PReLU slope

        elif self.activation_type == 'leaky_relu':
            nonlinearity = 'leaky_relu'
            slope_init = 0.01 # default nn.LeakyReLU slope

        elif self.apl is not None:
            nonlinearity = 'relu' # or 'random'
            slope_init = 0.

        elif self.deepspline is not None:

            if self.spline_init == 'prelu':
                nonlinearity = 'leaky_relu'
                slope_init = 0.25

            elif self.spline_init == 'leaky_relu':
                nonlinearity = 'leaky_relu'
                slope_init = 0.01

            else:
                nonlinearity = self.spline_init
                slope_init = 0.

        else:
            nonlinearity = self.activation_type
            slope_init = 0.

        nonlinearities = ['random', 'custom_normal', 'even_odd', 'identity', 'softplus', 'relu', 'leaky_relu']
        assert nonlinearity in nonlinearities, ('nonlinearity should be in '
                                                '[random, custom_normal, even_odd, identity, softplus, relu, leaky_relu].')


        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                if nonlinearity in ['random', 'even_odd', 'identity', 'softplus']:
                    nn.init.xavier_normal_(module.weight)

                elif nonlinearity == 'custom_normal':
                    module.weight.data.normal_(0, 0.05)
                    module.bias.data.zero_()

                elif control == 'forward':
                    nn.init.kaiming_normal_(module.weight, a=slope_init, mode='fan_in',
                                            nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_normal_(module.weight, a=slope_init, mode='fan_out',
                                            nonlinearity=nonlinearity)

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    ###########################################################################
    # Parameters


    def get_num_params(self):
        """ """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params


    def modules_deepspline(self):
        """ """
        for module in self.modules():
            if isinstance(module, self.deepspline):
                yield module


    def named_parameters_no_deepspline_apl(self, recurse=True):
        """ Named parameters of network, excepting deepspline parameters.
        """
        try:
            for name, param in self.named_parameters(recurse=recurse):
                deepspline_param = False
                apl_param = False
                # get all deepspline parameters
                if self.deepspline is not None:
                    for param_name in self.deepspline.parameter_names():
                        if name.endswith(param_name):
                            deepspline_param = True

                if self.apl is not None:
                    for param_name in self.apl.parameter_names():
                        if name.endswith(param_name):
                            apl_param = True

                if (deepspline_param is False) and (apl_param is False):
                    yield name, param

        except AttributeError:
            print('Not using deepspline or apl activations...')
            raise



    def named_parameters_deepspline(self, recurse=True):
        """ Named parameters (for optimizer) of deepspline activations.
        """
        try:
            for name, param in self.named_parameters(recurse=recurse):
                deepspline_param = False
                for param_name in self.deepspline.parameter_names():
                    if name.endswith(param_name):
                        deepspline_param = True

                if deepspline_param is True:
                    yield name, param

        except AttributeError:
            print('Not using deepspline activations...')
            raise



    def named_parameters_apl(self, recurse=True):
        """ Named parameters (for optimizer) of apl activations.
        """
        try:
            for name, param in self.named_parameters(recurse=recurse):
                apl_param = False
                for param_name in self.apl.parameter_names():
                    if name.endswith(param_name):
                        apl_param = True

                if apl_param is True:
                    yield name, param

        except AttributeError:
            print('Not using apl activations...')
            raise



    def parameters_no_deepspline_apl(self):
        """ """
        for name, param in self.named_parameters_no_deepspline_apl(recurse=True):
            yield param



    def parameters_deepspline(self):
        """ """
        for name, param in self.named_parameters_deepspline(recurse=True):
            yield param



    def parameters_apl(self):
        """ """
        for name, param in self.named_parameters_apl(recurse=True):
            yield param



    def parameters_batch_norm(self):
        """ """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                yield module.weight, module.bias



    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


    ############################################################################
    # Deepsplines: regularization and sparsification

    @property
    def weight_decay_regularization(self):
        """ boolean """
        return ((self.hyperparam_tuning is True and self.params['lmbda'] > 0) or \
                (self.params['weight_decay'] > 0))


    @property
    def apl_weight_decay_regularization(self):
        """ boolean """
        return (self.apl is not None and self.params['beta'] > 0)


    @property
    def tv_bv_regularization(self):
        """ boolean """
        return (self.deepspline is not None and self.params['lmbda'] > 0)


    def init_hyperparams(self):
        """ Initialize per layer hyperparameters based on constant lmbda,
        and the hyperparameter relationship for global minimizer. In this
        case, lmbda becomes a constant from which the TV/BV(2) and weight decay
        regularization weights are computed for each layer, and not the TV/BV(2)
        regularization weight directly.

        For more information, see Theorem 3 and 4 in the deep spline networks paper.
        """
        self.wd_hyperparam = [] # weight decay hyperparameter list
        self.apl_wd_hyperparam = [] # apl weight decay hyperparameter list
        self.tv_bv_hyperparam = [] # TV(2)/BV(2) hyperparameter list

        with torch.no_grad():

            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    if self.deepspline is None or self.hyperparam_tuning is False:
                        # in this case, the hyperparameters will be the same for all layers.
                        self.wd_hyperparam.append(self.params['weight_decay']/2)
                    else:
                        weight_norm_sq = module.weight.data.pow(2).sum().item()
                        self.wd_hyperparam.append(self.params['lmbda']/(2.0*weight_norm_sq))

                elif self.apl is not None and isinstance(module, self.apl):
                    self.apl_wd_hyperparam.append(self.params['beta']/2)

                elif self.deepspline is not None and isinstance(module, self.deepspline):
                    if self.hyperparam_tuning is False:
                        # in this case, the hyperparameters will be the same for all layers.
                        self.tv_bv_hyperparam.append(self.params['lmbda'])
                    else:
                        module_tv_bv = module.totalVariation()
                        if self.params['lipschitz'] is True:
                            module_tv_bv = module_tv_bv + module.fZerofOneAbs()

                        module_tv_bv_l1 = module_tv_bv.norm(p=1).item()
                        self.tv_bv_hyperparam.append(self.params['lmbda']/module_tv_bv_l1)


        if self.params['verbose']:
            print('\n\nHyperparameters:')
            print('\nwd hyperparam :', self.wd_hyperparam, sep='\n')
            print('\ntv/bv hyperparam :', self.tv_bv_hyperparam, sep='\n')



    def weight_decay(self):
        """ Computes the total weight decay of the network.

        For the resnet, also apply weight decay with a fixed
        value to the batchnorm weights and biases.
        Note: Fixed weight decay is also applied to the explicit linear
        parameters, if using DeepBSplineExplicitLinear activation.
        """
        wd = Tensor([0.]).to(self.device)

        i = 0
        for module in self.modules():
            if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    wd = wd + self.wd_hyperparam[i] * module.weight.pow(2).sum()
                    i += 1
                else:
                    wd = wd + self.params['weight_decay']/2 * module.weight.pow(2).sum()

            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                wd = wd + self.params['weight_decay']/2 * module.bias.pow(2).sum()

        assert i == len(self.wd_hyperparam)

        return wd[0] # 1-element 1d tensor -> 0d tensor



    def apl_weight_decay(self):
        """ Computes the sum of the weight decay of all apl activations
        """
        apl_wd = Tensor([0.]).to(self.device)
        tv_bv_unweighted = Tensor([0.]).to(self.device) # for printing loss without weighting

        i = 0
        for module in self.modules():
            if isinstance(module, self.apl):
                apl_wd = apl_wd + self.apl_wd_hyperparam[i] * module.a_apl.pow(2).sum()
                apl_wd = apl_wd + self.apl_wd_hyperparam[i] * module.b_apl.pow(2).sum()
                i += 1

        assert i == len(self.apl_wd_hyperparam)

        return apl_wd[0] # 1-element 1d tensor -> 0d tensor



    def TV_BV(self):
        """ Computes the sum of the TV(2)/BV(2) norm of all activations

        Returns:
            BV(2), if lipschitz is True;
            TV(2), if lipschitz is False.
        """
        tv_bv = Tensor([0.]).to(self.device)
        tv_bv_unweighted = Tensor([0.]).to(self.device) # for printing loss without weighting

        i = 0
        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_tv_bv = module.totalVariation(mode='additive')
                if self.params['lipschitz'] is True:
                    module_tv_bv = module_tv_bv + module.fZerofOneAbs(mode='additive')

                tv_bv = tv_bv + self.tv_bv_hyperparam[i] * module_tv_bv.norm(p=self.outer_norm)
                with torch.no_grad():
                    tv_bv_unweighted = tv_bv_unweighted + module_tv_bv.norm(p=self.outer_norm)
                i += 1

        assert i == len(self.tv_bv_hyperparam)

        return tv_bv[0], tv_bv_unweighted[0] # 1-element 1d tensor -> 0d tensor



    def lipschitz_bound(self):
        """ Returns the lipschitz bound of the network

        The lipschitz bound associated with C is:
        ||f_deep(x_1) - f_deep(x_2)||_1 <= C ||x_1 - x_2||_1,
        for all x_1, x_2 \in R^{N_0}.

        For l \in {1, ..., L}, n \in {1,..., N_l}:
        w_{n, l} is the vector of weights from layer l-1 to layer l, neuron n;
        s_{n, l} is the activation function of layer l, neuron n.

        C = (prod_{l=1}^{L} [max_{n,l} w_{n,l}]) * (prod_{l=1}^{L} ||s_l||_{BV(2)}),
        where ||s_l||_{BV(2)} = sum_{n=1}^{N_l} {TV(2)(s_{n,l}) + |s_{n,l}(0)| + |s_{n,l}(1)|}.

        For details, please see https://arxiv.org/pdf/2001.06263v1.pdf
        (Theorem 1, with p=1, q=\infty).
        """
        bv_product = Tensor([1.]).to(self.device)
        max_weights_product = Tensor([1.]).to(self.device)

        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_tv = module.totalVariation()
                module_fzero_fone = module.fZerofOneAbs()
                bv_product = bv_product * (module_tv.sum() + module_fzero_fone.sum())

            elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                max_weights_product = max_weights_product * module.weight.data.abs().max()

        lip_bound = max_weights_product * bv_product

        return lip_bound[0] # 1-element 1d tensor -> 0d tensor



    def sparsify_activations(self):
        """ Sparsifies the deepspline activations, eliminating the slopes
        smaller than a threshold.

        Note that deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        This function sets a_k to zero if |a_k| < slope_threshold.
        """
        for module in self.modules():
            if isinstance(module, self.deepspline):
                module.apply_threshold(self.slope_threshold)


    def compute_sparsity(self):
        """ Returns the sparsity of the activations (see deepspline.py)
        """
        sparsity = 0
        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_sparsity, _ = module.get_threshold_sparsity(self.slope_threshold)
                sparsity += module_sparsity.sum().item()

        return sparsity



    def get_deepspline_activations(self):
        """ Returns a list of activation parameters for each deepspline activation layer.
        """
        with torch.no_grad():
            activations_list = []
            for name, module in self.named_modules():

                if isinstance(module, self.deepspline):
                    grid_tensor = module.grid_tensor # (num_activations, size)
                    if module.mode == 'conv':
                        input = grid_tensor.transpose(0,1).unsqueeze(-1).unsqueeze(-1) # 4D

                    output = module(input)
                    if module.mode == 'conv':
                        # (num_activations, size)
                        output = output.transpose(0,1).squeeze(-1).squeeze(-1)

                    _, threshold_sparsity_mask = module.get_threshold_sparsity(self.slope_threshold)
                    activations_list.append({'name': '_'.join([name, module.mode]),
                                            'x': grid_tensor.clone().detach().cpu(),
                                            'y': output.clone().detach().cpu(),
                                            'threshold_sparsity_mask' : threshold_sparsity_mask.cpu()})

        return activations_list
