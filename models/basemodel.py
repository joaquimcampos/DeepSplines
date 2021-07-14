import torch
from torch import nn
from torch import Tensor

from models.deepBspline import DeepBSpline
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear
from models.deepRelu import DeepReLU
from ds_utils import spline_grid_from_range


class BaseModel(nn.Module):

    def __init__(self, **params):
        """ """
        super().__init__()

        self.params = params

        # general attributes
        self.set_attributes('activation_type', 'dataset_name',
                            'num_classes', 'device')
        # deepspline attributes
        self.set_attributes('spline_init', 'spline_size', 'spline_range',
                            'save_memory', 'knot_threshold')

        self.spline_grid = \
            spline_grid_from_range(self.spline_size, self.spline_range)

        self.deepspline = None
        if self.activation_type == 'deepBspline':
            self.deepspline = DeepBSpline
        elif self.activation_type == 'deepBspline_explicit_linear':
            self.deepspline = DeepBSplineExplicitLinear
        elif self.activation_type == 'deepRelu':
            self.deepspline = DeepReLU


    def set_attributes(self, *names):
        """ """
        assert hasattr(self, 'params'), 'self.params does not exist.'
        for name in names:
            assert isinstance(name, str), f'{name} is not string.'
            if name in self.params:
                setattr(self, name, self.params[name])


    ############################################################################
    # Activation initialization


    def init_activation_list(self, activation_specs, bias=False, **kwargs):
        """
        Initialize list of activation modules.

        Args:
            activation_specs (list):
                list of pairs ('layer_type', num_channels/neurons);
                len(activation_specs) = number of activation layers;
                e.g., [('conv', 64), ('fc', 100)].

            bias (bool):
                explicit bias;
                only relevant if self.deepspline == DeepBSplineExplicitLinear.

        Returns:
            activations (nn.ModuleList)
        """
        assert isinstance(activation_specs, list)

        if self.deepspline is not None:
            activations = nn.ModuleList()
            for mode, num_activations in activation_specs:
                activations.append(self.deepspline(size=self.spline_size, grid=self.spline_grid,
                                                init=self.spline_init, bias=bias, mode=mode,
                                                num_activations=num_activations,
                                                save_memory=self.save_memory, device=self.device))
        else:
            activations = self.init_standard_activations(activation_specs)

        return activations



    def init_activation(self, activation_specs, **kwargs):
        """
        Initialize a single activation module

        Args:
            activation_specs (tuple): e.g., ('conv', 64)

        Returns:
            activation (nn.Module)
        """
        assert isinstance(activation_specs, tuple)
        activation = self.init_activation_list([activation_specs], **kwargs)[0]

        return activation



    def init_standard_activations(self, activation_specs, **kwargs):
        """
        Initialize non-spline activation modules.

        Args:
            activation_type :
                'relu', 'leaky_relu'.

            activation_specs :
                list of pairs ('layer_type', num_channels/neurons);
                len(activation_specs) = number of activation layers.
                e.g., [('conv', 64), ('fc', 100)].

        Returns:
            activations (nn.ModuleList)
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

        else:
            raise ValueError(f'{self.activation_type} is not in relu family...')

        return activations



    def initialization(self, init_type='He'):
        """
        Initializes the network weights with 'He', 'Xavier', or a
        custom gaussian initialization.
        """
        assert init_type in ['He', 'Xavier', 'custom_normal']

        if init_type == 'He':
            if self.activation_type in ['leaky_relu', 'relu']:
                nonlinearity = self.activation_type
                slope_init = 0.01 if nonlinearity == 'leaky_relu' else 0.

            elif self.deepspline is not None and self.spline_init in ['leaky_relu', 'relu']:
                nonlinearity = self.spline_init
                slope_init = 0.01 if nonlinearity == 'leaky_relu' else 0.
            else:
                init_type = 'Xavier' # overwrite init_type


        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                if init_type == 'Xavier':
                    nn.init.xavier_normal_(module.weight)

                elif init_type == 'custom_normal':
                    # custom Gauss(0, 0.05) weight initialization
                    module.weight.data.normal_(0, 0.05)
                    module.bias.data.zero_()

                else: # He initialization
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
        """
        Yields all deepspline modules in the network.
        """
        for module in self.modules():
            if isinstance(module, self.deepspline):
                yield module


    def named_parameters_no_deepspline(self, recurse=True):
        """
        Yields all named_parameters in the network,
        excepting deepspline parameters.
        """
        try:
            for name, param in self.named_parameters(recurse=recurse):
                deepspline_param = False
                # get all deepspline parameters
                if self.deepspline is not None:
                    for param_name in self.deepspline.parameter_names():
                        if name.endswith(param_name):
                            deepspline_param = True

                if deepspline_param is False:
                    yield name, param

        except AttributeError:
            print('Not using deepspline activations...')
            raise



    def named_parameters_deepspline(self, recurse=True):
        """
        Yields all deepspline named_parameters in the network.
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



    def parameters_no_deepspline(self):
        """
        Yields all parameters in the network,
        excepting deepspline parameters.
        """
        for name, param in self.named_parameters_no_deepspline(recurse=True):
            yield param



    def parameters_deepspline(self):
        """
        Yields all deepspline parameters in the network.
        """
        for name, param in self.named_parameters_deepspline(recurse=True):
            yield param



    def parameters_batch_norm(self):
        """
        Yields all batch_norm parameters in the network.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                yield module.weight, module.bias



    def freeze_parameters(self):
        """
        Freezes the network (no gradient computations).
        """
        for param in self.parameters():
            param.requires_grad = False


    ############################################################################
    # Deepsplines: regularization and sparsification

    @property
    def weight_decay_regularization(self):
        """
        Flag indicating whether weight_decay is applied.
        """
        return (self.params['weight_decay'] > 0)


    @property
    def tv_bv_regularization(self):
        """
        Flag indicating whether TV2/BV2 regularization is applied.
        """
        return (self.deepspline is not None and self.params['lmbda'] > 0)



    def weight_decay(self):
        """
        Computes the total weight decay of the network.

        Returns:
            wd (float):
                = mu/2 * (sum(weights^2) + sum(biases^2))
        """
        wd = Tensor([0.]).to(self.device)

        for module in self.modules():
            if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                wd = wd + self.params['weight_decay']/2 * module.weight.pow(2).sum()

            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                wd = wd + self.params['weight_decay']/2 * module.bias.pow(2).sum()

        return wd[0] # 1-tap 1d tensor -> 0d tensor



    def TV_BV(self):
        """
        Computes the sum of the TV(2)/BV(2) norm of all
        deepspline activations in the network.

        Returns:
            tv_bv (float):
                = lambda*sum(BV(2)) if lipschitz is True, otherwise = lambda*sum(TV(2))
            tv_bv_unweighted (float):
                = sum(BV(2)) if lipschitz is True, otherwise = sum(TV(2))
        """
        tv_bv = Tensor([0.]).to(self.device)
        # for printing loss without weighting
        tv_bv_unweighted = Tensor([0.]).to(self.device)

        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_tv_bv = module.totalVariation(mode='additive')
                if self.params['lipschitz'] is True:
                    module_tv_bv = module_tv_bv + module.fZerofOneAbs(mode='additive')

                tv_bv = tv_bv + self.params['lmbda'] * module_tv_bv.norm(p=1)
                with torch.no_grad():
                    tv_bv_unweighted = tv_bv_unweighted + module_tv_bv.norm(p=1)

        return tv_bv[0], tv_bv_unweighted[0] # 1-tap 1d tensor -> 0d tensor



    def lipschitz_bound(self):
        """
        Computes the lipschitz bound of the network

        The lipschitz bound associated with C is:
        ||f_net(x_1) - f_net(x_2)||_1 <= C ||x_1 - x_2||_1,
        for all x_1, x_2 \in R^{N_0}.

        For l \in {1, ..., L}, n \in {1,..., N_l}:
        w_{n, l} is the vector of weights from layer l-1 to layer l, neuron n;
        s_{n, l} is the activation function of layer l, neuron n.

        C = (prod_{l=1}^{L} [max_{n,l} w_{n,l}]) * (prod_{l=1}^{L} ||s_l||_{BV(2)}),
        where ||s_l||_{BV(2)} = sum_{n=1}^{N_l} {BV(2)(s_{n,l})}.

        Returns:
            lip_bound (float):
                global lipschitz bound of the network
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

        return lip_bound[0] # 1-tap 1d tensor -> 0d tensor



    def sparsify_activations(self):
        """
        Sparsifies the deepspline activations, eliminating slope
        changes (a_k) smaller than a threshold.

        Note that deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        This function sets a_k to zero if |a_k| < knot_threshold.
        """
        for module in self.modules():
            if isinstance(module, self.deepspline):
                module.apply_threshold(self.knot_threshold)


    def compute_sparsity(self):
        """
        Returns the sparsity of the activations,
        i.e. the number of activation knots (see deepspline.py).

        Returns:
            sparsity (int)
        """
        sparsity = 0
        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_sparsity, _ = module.get_threshold_sparsity(self.knot_threshold)
                sparsity += module_sparsity.sum().item()

        return sparsity



    def get_deepspline_activations(self):
        """
        Get information of deepspline activations (e.g. for plotting).

        Returns:
            activations_list (list):
                Length = number of deepspline activations.
                Each entry is a  dictionary of the form
                {'name': activation name,
                  'locations': position of the B-spline basis,
                  'coefficients': deepspline coefficients,
                  'threshold_sparsity_mask': mask indicating (non-zero) knots}
        """
        with torch.no_grad():
            activations_list = []
            for name, module in self.named_modules():

                if isinstance(module, self.deepspline):
                    locations = module.grid_tensor # (num_activations, size)
                    input = locations.transpose(0,1) # (size, num_activations)
                    if module.mode == 'conv':
                        input = input.unsqueeze(-1).unsqueeze(-1) # to 4D

                    output = module(input)
                    coefficients = output.transpose(0, 1)
                    if module.mode == 'conv':
                        coefficients = coefficients.squeeze(-1).squeeze(-1) # to 2D
                    # coefficients: (num_activations, size)

                    _, threshold_sparsity_mask = module.get_threshold_sparsity(self.knot_threshold)
                    activations_list.append({'name': '_'.join([name, module.mode]),
                                            'locations': locations.clone().detach().cpu(),
                                            'coefficients': coefficients.clone().detach().cpu(),
                                            'sparsity_mask' : threshold_sparsity_mask.cpu()})

        return activations_list
