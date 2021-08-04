
.. image:: https://user-images.githubusercontent.com/26142730/128066373-a42476b4-6694-4810-8397-d6e1fa2638a8.png
  :width: 50 %
  :align: center

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5158821.svg
   :target: https://doi.org/10.5281/zenodo.5158821


*DeepSplines* is a framework to train the activation functions of a neural network.

The aim of this repository is to:

* Facilitate the reproduction of the results reported in the research papers:

  * `Learning Activation Functions in Deep (Spline) Neural Networks <http://bigwww.epfl.ch/publications/bohra2003.html>`_;

  * `Deep Neural Networks with Trainable Activations and Controlled Lipschitz Constant <http://bigwww.epfl.ch/publications/aziznejad2001.html>`_.

* Enable a seamless integration of deep spline activation functions in a custom neural network.


The proposed scheme is based on the theoretical work of
`M.Unser <http://bigwww.epfl.ch/publications/unser1901.html>`_.


.. contents:: **Table of Contents**
    :depth: 2


Requirements
============

* python >= 3.7
* numpy >= 1.10
* pytorch >= 1.5.1
* torchvision >= 0.2.2
* matplotlib >= 3.3.1
* (optional) CUDA

The code was developed and tested on a x86_64 Linux system.

Installation
============

To install the package, we first create an environment with python 3.7 (or greater):

.. code-block:: bash

    >> conda create -y -n deepsplines python=3.7
    >> source activate deepsplines

Quick Install
--------------

*DeepSplines* is available on `Pypi <https://pypi.org/project/deepsplines/>`_.
Therefore, you can install the package via the command:

.. code-block:: bash

    >> pip install deepsplines

.. role:: bash(code)
   :language: bash

For NVIDIA GPU compatibility, you need to additionally install :bash:`cudatoolkit`
(via :bash:`conda install -c anaconda cudatoolkit`)

Developper Install
------------------

It is also possible to install *DeepSplines* from the source for developpers:

.. code-block:: bash

   >> git clone https://github.com/joaquimcampos/DeepSplines
   >> cd <repository_dir>/
   >> pip install -e .

Usage
=====

Here we show an example on how to adapt the `PyTorch CIFAR-10 tutorial <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_
to use DeepBSpline activations.

.. code-block:: python

    from deepsplines.ds_modules import dsnn


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


For full details, please consult `scripts/deepsplines_tutorial.py <https://github.com/joaquimcampos/DeepSplines/blob/master/scripts/deepsplines_tutorial.py>`_.

Reproducing results
-------------------

To reproduce the results shown in the research papers [Bohra-Campos2020]_ and [Aziznejad2020]_ one can run the following scripts:

.. code-block:: bash

    >> ./scripts/run_resnet32_cifar.py
    >> ./scripts/run_nin_cifar.py
    >> ./scripts/run_twoDnet.py

To see the running options, please add :bash:`--help` to the commands above.

Developers
==========

*DeepSplines* is developed by the `Biomedical Imaging Group <http://bigwww.epfl.ch/>`_,
`École Polytéchnique Fédérale de Lausanne <https://www.epfl.ch/en/>`_, Switzerland.

For citing this package, please see: http://doi.org/10.5281/zenodo.5156042

Original authors:

-   **Joaquim Campos** (joaquimcampos15@hotmail.com)
-   **Pakshal Bohra** (pakshal.bohra@epfl.ch)

Contributors:

-   **Harshit Gupta**

References
==========

.. [Bohra-Campos2020] P. Bohra, J. Campos, H. Gupta, S. Aziznejad, M. Unser, "Learning Activation Functions in Deep (Spline) Neural Networks," IEEE Open Journal of Signal Processing, vol. 1, pp.295-309, November 19, 2020.

.. [Aziznejad2020] S. Aziznejad, H. Gupta, J. Campos, M. Unser, "Deep Neural Networks with Trainable Activations and Controlled Lipschitz Constant," IEEE Transactions on Signal Processing, vol. 68, pp. 4688-4699, August 10, 2020.

License
=======

The code is released under the terms of the `MIT License <https://github.com/joaquimcampos/DeepSplines/blob/master/LICENSE>`_

Acknowledgements
================

This work was supported in part by the Swiss National Science Foundation under Grant 200020_184646 / 1 and in part by the European Research Council (ERC)
under Grant 692726-GlobalBioIm.
