""" Wrap around nn.Module to accomodate for Deepsplines """


import torch
from torch import nn
from torch import Tensor

from models.deepBspline import DeepBSpline
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear
from models.deepRelu import DeepReLU
from ds_utils import spline_grid_from_range


class BaseModel(nn.Module):

    def __init__(self, activation_type=None, dataset_name=None,
                num_classes=None, device=None,
                spline_init=None, spline_size=None,
                spline_range=None, save_memory=False,
                knot_threshold=0., **kwargs):
        """
        Args:
            ------ general -----------------------

            activation_type (str):
                'relu', 'leaky_relu', 'deepBspline',
                'deepBspline_explicit_linear', or 'deepRelu'.
            dataset_name (str):
                s_shape_1500', 'circle_1500', 'cifar10', 'cifar100' or 'mnist'.
            num_classes (int):
                number of dataset classes.
            device (str):
                'cuda:0' or 'cpu'

            ------ deepspline --------------------

            spline_init (str):
                Function to initialize activations as (e.g. 'leaky_relu').
            spline_size (odd int):
                number of coefficients of spline grid;
                the number of knots K = size - 2.
            spline_range (float):
                Defines the range of the B-spline expansion;
                B-splines range = [-spline_range, spline_range].
            save_memory (bool):
                If true, use a more memory efficient version (takes more time);
                Can be used only with deepBsplines.
                (see deepBspline_base.py docstring for details.)
            knot_threshold (bool):
                If nonzero, sparsify activations by eliminating knots whose
                slope change is below this value.
        """
        # TODO: Check default values of arguments

        super().__init__()

        self.params = params

        # general attributes
        self.activation_type = activation_type
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.device = device

        # deepspline attributes
        self.spline_init = spline_init
        self.spline_size = spline_size
        self.spline_range = spline_range
        self.save_memory = save_memory
        self.knot_threshold = knot_threshold

        self.spline_grid = \
            spline_grid_from_range(self.spline_size, self.spline_range)

        self.deepspline = None
        if self.activation_type == 'deepBspline':
            self.deepspline = DeepBSpline
        elif self.activation_type == 'deepBspline_explicit_linear':
            self.deepspline = DeepBSplineExplicitLinear
        elif self.activation_type == 'deepRelu':
            self.deepspline = DeepReLU


    ############################################################################
    # Activation initialization


    def init_activation_list(self, activation_specs, bias=True, **kwargs):
        """
        Initialize list of activation modules.

        Args:
            activation_specs (list):
                list of pairs ('layer_type', num_channels/neurons);
                len(activation_specs) = number of activation layers;
                e.g., [('conv', 64), ('fc', 100)].

            bias (bool):
                if True, add explicit bias to deepspline;
                only relevant if self.deepspline == DeepBSplineExplicitLinear.

        Returns:
            activations (nn.ModuleList)
        """
        assert isinstance(activation_specs, list)

        if self.deepspline is not None:
            activations = nn.ModuleList()
            for mode, num_activations in activation_specs:
                activations.append(self.deepspline(mode=mode, size=self.spline_size,
                                                grid=self.spline_grid, init=self.spline_init,
                                                bias=bias, num_activations=num_activations,
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
    def using_deepsplines(self):
        """
        True if using deepspline activations.
        """
        return (self.deepspline is not None)



    def l2sqsum_weights_biases(self):
        """
        Computes the sum of the l2 squared norm of the weights and biases
        of the network.

        Returns:
            l2sqsum (0d Tensor):
                l2sqsum = (sum(weights^2) + sum(biases^2))
        """
        wd = Tensor([0.]).to(self.device)

        for module in self.modules():
            if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                l2sqsum = l2sqsum + module.weight.pow(2).sum()

            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                l2sqsum = l2sqsum + module.bias.pow(2).sum()

        return l2sqsum[0] # 1-tap 1d tensor -> 0d tensor



    def TV_BV(self):
        """
        Computes the sum of the TV(2)/BV(2) norm of all
        deepspline activations in the network.

        Returns:
            tv_bv (0d Tensor):
                tv_bv = sum(BV(2)) if lipschitz is True, otherwise = sum(TV(2))
        """
        tv_bv = Tensor([0.]).to(self.device)

        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_tv_bv = module.totalVariation(mode='additive')
                if self.params['lipschitz'] is True:
                    module_tv_bv = module_tv_bv + module.fZerofOneAbs(mode='additive')

                tv_bv = tv_bv + module_tv_bv.norm(p=1)

        return tv_bv[0] # 1-tap 1d tensor -> 0d tensor



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
            lip_bound (0d Tensor):
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
