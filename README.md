<img src="https://github.com/joaquimcampos/DeepSplines/blob/master/data/deepspline_basis.png" width=50% height=50%>

# Deep Spline Neural Networks

DeepSplines is a framework to train the activation functions of a neural network.

The aim of this repository is to:

-   Facilitate the reproduction of the results reported in the research papers
       -   [Learning Activation Functions in Deep (Spline) Neural Networks](http://bigwww.epfl.ch/publications/bohra2003.html) [[1]](#1)   
       -   [Deep Neural Networks with Trainable Activations and Controlled Lipschitz Constant](http://bigwww.epfl.ch/publications/aziznejad2001.html) [[2]](#2)
-   Enable a seamless integration of Deep Spline activation functions in
    a custom neural network.


The proposed scheme is based on the theoretical work of
[M. Unser](http://bigwww.epfl.ch/publications/unser1901.html) [[3]](#3).


2.  [Installation](#org2)
3.  [Usage](#org3)
    1.  [Reproducing results](#org31)
4.  [Authors and contributors](#org4)


<a id="org2"></a>
# Installation

A minimal installation requires:

-   numpy >= 1.10
-   pytorch >= 1.5.1
-   torchvision >= 0.2.2
-   matplotlib >= 3.3.1
-   CUDA

<!-- These requirements can be installed using conda (replace `<X.X>` by your
CUDA version)

    conda create -y -n deepsplines python=3.8 cudatoolkit=<X.X>
    pip install git+https://github.com/joaquimcampos/DeepSplines@develop
    source activate deepsplines -->

To use DeepSplines with PyTorch install:

    conda create -y -n deepsplines python=3.8 cudatoolkit=<X.X> pytorch \
          -c defaults -c pytorch -c conda-forge
    source activate deepsplines
    # Install development version of deepsplines:
    pip install git+https://github.com/joaquimcampos/DeepSplines@develop


<a id="org3"></a>
# Usage

Example on how to adapt the [PyTorch CIFAR-10 tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
to use DeepBSpline activations.

```
from deepsplines.modules import dsnn


class DSNet(dsnn.DSModule):
    def __init__(self):

        super().__init__()

        self.conv_ds = nn.ModuleList()
        self.fc_ds = nn.ModuleList()

        # deepspline parameters
        opt_params = {
            'size': 51,
            'range_': 4,
            'init': 'leaky_relu',
            'save_memory': False
        }

        # convolutional layer with 6 output channels
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv_ds.append(dsnn.DeepBSpline('conv', 6, **opt_params))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv_ds.append(dsnn.DeepBSpline('conv', 16, **opt_params))

        # fully-connected layer with 120 output units
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_ds.append(dsnn.DeepBSpline('fc', 120, **opt_params))
        self.fc2 = nn.Linear(120, 84)
        self.fc_ds.append(dsnn.DeepBSpline('fc', 84, **opt_params))
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.pool(self.conv_ds[0](self.conv1(x)))
        x = self.pool(self.conv_ds[1](self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc_ds[0](self.fc1(x))
        x = self.fc_ds[1](self.fc2(x))
        x = self.fc3(x)

        return x

dsnet = DSNet()
dsnet.to(device)

main_optimizer = optim.SGD(dsnet.parameters_no_deepspline(),
                           lr=0.001,
                           momentum=0.9)
aux_optimizer = optim.Adam(dsnet.parameters_deepspline())

lmbda = 1e-4 # regularization weight
lipschitz = False # lipschitz control

for epoch in range(2):

    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        main_optimizer.zero_grad()
        aux_optimizer.zero_grad()

        outputs = dsnet(inputs)
        loss = criterion(outputs, labels)

        # add regularization loss
        if lipschitz is True:
            loss = loss + lmbda * dsnet.BV2()
        else:
            loss = loss + lmbda * dsnet.TV2()

        loss.backward()
        main_optimizer.step()
        aux_optimizer.step()
```

For full details, please consult the [tutorial](scripts/deepsplines_tutorial.py).

<a id="org31"></a>
## Reproducing results

To reproduce the results shown in the research papers [[1]](#2) and [[2]](#2)
one can run the following scripts:

-    ``./deepsplines/scripts/run_resnet32_cifar.py``
-    ``./deepsplines/scripts/run_nin_cifar.py``
-    ``./deepsplines/scripts/run_twoDnet.py``

To see the running options, please add ``--help`` to the commands above.

<a id="org4"></a>
# Authors and contributors

DeepSplines is developed by the Biomedical Imaging Group at BIG. Original authors:

-   **Joaquim Campos**
-   **Pakshal Bohra**

Contributor:
-   **Harshit Gupta**


# References
<a id="1">[1]</a>
P. Bohra, J. Campos, H. Gupta, S. Aziznejad, M. Unser,
"Learning Activation Functions in Deep (Spline) Neural Networks,"
IEEE Open Journal of Signal Processing, vol. 1, pp.295-309, November 19, 2020.

<a id="2">[2]</a>
S. Aziznejad, H. Gupta, J. Campos, M. Unser,
"Deep Neural Networks with Trainable Activations and Controlled Lipschitz Constant,"
IEEE Transactions on Signal Processing, vol. 68, pp. 4688-4699, August 10, 2020.

<a id="3">[3]</a>
M. Unser,
"A Representer Theorem for Deep Neural Networks,"
Journal of Machine Learning Research, vol. 20, no. 110, pp. 1-30, January 2019-Present.
