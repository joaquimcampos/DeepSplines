#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script exemplifies how to use DeepSplines in a network,
starting from the PyTorch CIFAR-10 tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Need to import dsnn (takes the role of torch.nn for DeepSplines)
from deepsplines.ds_modules import dsnn


########################################################################
# ReLU network


class Net(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


########################################################################
# Deepspline network

# We now show how to replace the ReLUs with DeepSpline activations in
# the previous network.

# we can use deepspline modules of three types:
# - DeepBspline
# - DeepBSplineExplicitLinear
# - DeepReLUSpline
# In this tutorial, we use the first as an example.

# The model needs to inherit from dsnn.DSModule. This is a wrap around
# nn.Module that contains all the DeepSpline functionality.


class DSNet(dsnn.DSModule):
    def __init__(self):

        super().__init__()

        # we put the deepsplines (ds) of the convolutional and fully-connected
        # layers in two separate nn.ModuleList() for simplicty.
        self.conv_ds = nn.ModuleList()
        self.fc_ds = nn.ModuleList()

        # We define some optional parameters for the deepspline
        # (see DeepBSpline.__init__())
        opt_params = {
            'size': 51,
            'range_': 4,
            'init': 'leaky_relu',
            'save_memory': False
        }

        self.conv1 = nn.Conv2d(3, 6, 5)
        # 1st parameter (mode): 'conv' (convolutional) / 'fc' (fully-connected)
        # 2nd parameter: nb. channels (mode='conv') / nb. neurons (mode='fc')
        self.conv_ds.append(dsnn.DeepBSpline('conv', 6, **opt_params))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv_ds.append(dsnn.DeepBSpline('conv', 16, **opt_params))

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_ds.append(dsnn.DeepBSpline('fc', 120, **opt_params))
        self.fc2 = nn.Linear(120, 84)
        self.fc_ds.append(dsnn.DeepBSpline('fc', 84, **opt_params))
        self.fc3 = nn.Linear(84, 10)

        self.initialization(opt_params['init'], init_type='He')
        self.num_params = self.get_num_params()

    def forward(self, x):

        x = self.pool(self.conv_ds[0](self.conv1(x)))
        x = self.pool(self.conv_ds[1](self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc_ds[0](self.fc1(x))
        x = self.fc_ds[1](self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":

    ########################################################################
    # Load the data

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    ########################################################################
    # Network, optimizer, loss

    net = Net()  # relu network
    net.to(device)
    print('ReLU: nb. parameters - {:d}'.format(net.num_params))

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    dsnet = DSNet()  # deepsplines network
    dsnet.to(device)
    print('DeepSpline: nb. parameters - {:d}'.format(dsnet.num_params))

    # For the parameters of the deepsplines, an optimizer different from "SGD"
    # is usually required for stability during training (Adam is recommended).
    # Therefore, when using an SGD optimizer for the network parameters, we
    # require an auxiliary one for the deepspline parameters.
    # Inherenting from DSModule allows us to use the parameters_deepspline()
    # and parameters_no_deepspline() methods for this.
    main_optimizer = optim.SGD(dsnet.parameters_no_deepspline(),
                               lr=0.001,
                               momentum=0.9)
    aux_optimizer = optim.Adam(dsnet.parameters_deepspline())

    criterion = nn.CrossEntropyLoss()

    ########################################################################
    # Training the ReLU network

    print('\nTraining ReLU network.')

    start_time = time.time()

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    end_time = time.time()

    print('Finished Training ReLU network. \n'
          'Took {:d} seconds. '.format(int(end_time - start_time)))

    ########################################################################
    # Training the DeepSpline network
    # Note: Since the original network is small, the time it takes to train
    # deepsplines is significantly larger.

    # Regularization weight for the TV(2)/BV(2) regularization
    # Needs to be tuned for performance
    lmbda = 1e-4
    # lipschitz control: if True, BV(2) regularization is used instead of TV(2)
    lipschitz = False

    print('\nTraining DeepSpline network.')

    start_time = time.time()

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            main_optimizer.zero_grad()
            aux_optimizer.zero_grad()

            # forward + backward + optimize
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

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    end_time = time.time()

    print('Finished Training DeepSpline network. \n'
          'Took {:d} seconds. '.format(int(end_time - start_time)))

    ########################################################################
    # Testing the ReLU and DeepSpline networks

    for model, name in zip([net, dsnet], ['ReLU', 'DeepSpline']):

        print(f'\nTesting {name} network.')

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients
        # for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose
                # as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the {name} network '
              'on the 10000 test images: %d %%' % (100 * correct / total))
