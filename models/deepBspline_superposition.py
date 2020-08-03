import torch
from torch import nn

from models.deepBspline import DeepBSpline
from models.deepBspline_explicit_linear import DeepBSplineExplicitLinear
from ds_utils import isPowerOfTwo



class DeepBSplineSuperposition(nn.Module):
    """
    Args:
        size : list of number of coefficients of spline grid (odd numbers)
        grid : list of spacings of spline knots
        init : Initialization of activation (type: str)
        bias : (flag) learn bias in explicit_linear activation (default: True)
    """
    def __init__(self, size=[51], grid=[0.1], init='leaky_relu',
                bias=True, **kwargs):

        # check that size list is increasing, only has odd elements
        # and is made of grid powers of two
        try:
            assert all(i % 2 == 1 for i in size)
        except AssertionError:
            raise ValueError(f'size list {size} has an even element...')
        try:
            assert all(i < j for i, j in zip(size, size[1:]))
        except AssertionError:
            raise ValueError(f'size list {size} is not increasing...')
        try:
            assert all(isPowerOfTwo((j//2)/(i//2)) for i, j in zip(size, size[1:]))
        except AssertionError:
            raise ValueError(f'size list {size} is not made of grid powers of two...')

        assert all(i > j for i, j in zip(grid, grid[1:])), f'{grid} is not decreasing...'

        super().__init__()
        self.size = size
        self.grid = grid
        self.init = init

        # first deepspline is initialized as self.init, the others are
        # initialized to 'zero' (overall superposition has correct Initialization)
        self.spline_superposition = nn.ModuleList() # deepspline
        self.spline_superposition.append(DeepBSplineExplicitLinear(size=size[0], grid=grid[0],
                                                                init=init, bias=bias, **kwargs))
        for i in range(1, len(size)):
            self.spline_superposition.append(DeepBSpline(size=size[i], grid=grid[i],
                                                        init='zero', **kwargs))

        # steps for index matching of lower resolution in higher resolution
        self.idx_steps = [((size[-1]//2)/(i//2)) for i in size]



    @staticmethod
    def parameter_names(**kwargs):
        """ Iterator over union of deepBspline and
        deepBspline_explicit_linear parameters names.
        """
        all_names = []
        for name in DeepBSpline.parameter_names():
            all_names.append(name)
            yield name
        for name in DeepBSplineExplicitLinear.parameter_names():
            if name not in all_names:
                yield name


    @property
    def grid_tensor(self):
        """ Return grid tensor of highest resolution.
        """
        return self.spline_superposition[0].grid_tensor

    @property
    def mode(self):
        """ """
        return self.spline_superposition[0].mode

    @property
    def num_activations(self):
        """ """
        return self.spline_superposition[0].num_activations

    @property
    def device(self):
        """ """
        return self.spline_superposition[0].device

    @property
    def bias(self):
        return self.spline_superposition[0].bias

    @property
    def weight(self):
        return self.spline_superposition[0].weight



    def forward(self, input):
        """
        Args:
            x : 4D input
        """
        output = self.spline_superposition[0](input)
        for i in range(1, len(self.size)):
            output = output + self.spline_superposition[i](input)

        return output



    def totalVariation(self, mode='true'):
        """ Check totalVariation() in deepBspline_base.py

        Args:
            mode='true' if computing the true tv value;
            mode='additive' if adding the tv of each individual
                superposed activation.
        """
        assert mode in ['true', 'additive']

        if mode == 'true':
            slopes = self.spline_superposition[-1].slopes
            for i in reversed(range(0, len(self.size)-1)):
                idx = torch.arange(0, self.size[-1], self.idx_steps[i])[1:-1].sub(1).long()
                slopes[:, idx] = slopes[:, idx] + self.spline_superposition[i].slopes

            tv = slopes.norm(1, dim=1)
        else:
            tv = self.spline_superposition[-1].totalVariation()
            for i in reversed(range(0, len(self.size)-1)):
                tv = tv + self.spline_superposition[i].totalVariation()

        return tv



    def get_threshold_sparsity(self, threshold):
        """ """
        _, threshold_sparsity_mask = self.spline_superposition[-1].get_threshold_sparsity(threshold)

        for i in reversed(range(0, len(self.size)-1)):
            idx = torch.arange(0, self.size[-1], self.idx_steps[i])[1:-1].sub(1).long()
            _, t_sparsity_mask = self.spline_superposition[i].get_threshold_sparsity(threshold)
            threshold_sparsity_mask[:, idx] = threshold_sparsity_mask[:, idx] | t_sparsity_mask

        threshold_sparsity = threshold_sparsity_mask.sum(dim=1)

        return threshold_sparsity, threshold_sparsity_mask



    def apply_threshold(self, threshold):
        """ see DeepBSplineBase.apply_threshold
        """
        for i in range(0, len(self.size)):
            self.spline_superposition[i].apply_threshold(threshold)



    def fZerofOneAbs(self, mode='true'):
        """ Computes the lipschitz regularization: |f(0)| + |f(1)|

        Args:
            mode='true' if computing the correct lipschitz value;
            mode='additive' if adding the lipschitz of each individual
                superposed activation.
        """
        assert mode in ['true', 'additive']

        if mode == 'true':
            fzero_fone = self.spline_superposition[-1].fZerofOne()
            for i in reversed(range(0, len(self.size)-1)):
                fzero_fone = fzero_fone + self.spline_superposition[i].fZerofOne()

            fzero_fone_abs = fzero_fone.abs().sum(0) # (2, num_activations)->(num_activations,)
        else:
            fzero_fone_abs = self.spline_superposition[-1].fZerofOneAbs()
            for i in reversed(range(0, len(self.size)-1)):
                fzero_fone_abs = fzero_fone_abs + self.spline_superposition[i].fZerofOneAbs()


        return fzero_fone_abs



    def extra_repr(self):
        """ repr for print(model)
        """
        s = ('mode={mode}, num_activations={num_activations}, init={init}, '
            'size={size}, grid={grid}, bias={learn_bias}.')

        return s.format(**self.__dict__)
