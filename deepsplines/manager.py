# References :
# https://github.com/kuangliu/pytorch-cifar
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
Module for managing the training and testing.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from deepsplines.dataloader import DataLoader
from deepsplines.project import Project
from deepsplines.ds_utils import update_running_losses

from deepsplines.networks import (TwoDNet, ResNet32Cifar, NiNCifar,
                                  ConvNetMnist)
from deepsplines.datasets import init_dataset

##########################################################################
# MANAGER


class Manager(Project):
    """ Class that manages training and testing """
    def __init__(self, params, user_params):
        """
        Args:
            params (dict):
                dictionary with all parameters.
            user_params (dict):
                dictionary with the user-defined parameters.
                Used to override saved parameters from a checkpoint
                if continuing training or testing a model.
        """

        super().__init__(params, user_params)

        is_ckpt_loaded = False
        if self.load_ckpt is True:  # resuming training or testing
            # is_ckpt_loaded=True if a checkpoint was successfully loaded.
            is_ckpt_loaded = self.restore_ckpt_params()

        self.init_device()  # initalize device (e.g. 'cpu', 'cuda')
        self.init_log()  # initalize log directory

        if self.params['verbose']:
            print('\n==> Parameters: ', self.params, sep='\n')

        self.dataset = init_dataset(**self.params['dataset'])

        # model requires number of dataset classes for the output layer.
        self.params['model']['num_classes'] = self.dataset.num_classes
        self.net = self.build_model(self.params, self.device)

        if self.training:
            self.set_optimization()

        if is_ckpt_loaded is True:
            self.restore_model()

        self.init_json()

        # During testing, average the loss only at the end to get accurate
        # value of the loss per sample. If using reduction='mean', when
        # nb_test_samples % batch_size != 0 we can only average the loss per
        # batch (as in training for printing the losses) but not per sample.
        if self.params['net'] == 'twoDnet':
            self.criterion = nn.BCELoss(reduction='mean')
            self.test_criterion = nn.BCELoss(reduction='sum')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.test_criterion = nn.CrossEntropyLoss(reduction='sum')

        self.criterion.to(self.device)
        self.test_criterion.to(self.device)

        # # uncomment for printing network architecture
        # print(self.net)

    @staticmethod
    def build_model(params, device='cuda:0'):
        """
        Build the network model.

        Args:
            params (dict):
                contains the network name and the model parameters.
            device (str):
                'cpu' or 'cuda:0'.
        Returns:
            net (nn.Module)
        """
        print('\n==> Building model...')

        networks_dict = {
            'twoDnet': TwoDNet,
            'resnet32_cifar': ResNet32Cifar,
            'nin_cifar': NiNCifar,
            'convnet_mnist': ConvNetMnist
        }

        assert params['net'] in networks_dict.keys(), \
            'network not found: please add net to networks_dict.'
        net = networks_dict[params['net']](**params['model'])

        net = net.to(device)
        if device.startswith('cuda'):
            cudnn.benchmark = True

        print(f'[Network] Total number of parameters : {net.num_params}.')

        return net

    def set_optimization(self):
        """
        Sets the training optimizers and learning rate schedulers.

        If self.params['optimizer'] has a single element, then
        a single optimizer/scheduler is used for the network and
        the deepspline parameters.

        If self.params['optimizer'] is a list of two elements, the first
        element is the 'main' optimizer (for the network parameters) and the
        second 'aux' optimizer (for the deepspline parameters).

        Note: An 'aux' optimizer different from SGD is usually required
        for training deepsplines. Adam generally works well.
        """
        self.optim_names = self.params['optimizer']

        # main optimizer/scheduler
        if len(self.optim_names) == 2:
            try:
                # main optimizer only for network parameters
                main_params_iter = self.net.parameters_no_deepspline()
            except AttributeError:
                print('Cannot use aux optimizer.')
                raise
        else:
            # single optimizer for all parameters
            main_params_iter = self.net.parameters()

        self.main_optimizer = self.construct_optimizer(main_params_iter,
                                                       self.optim_names[0],
                                                       self.params['lr'])

        self.main_scheduler = self.construct_scheduler(self.main_optimizer)

        self.aux_optimizer, self.aux_scheduler = None, None

        if len(self.optim_names) == 2:
            # aux optimizer/scheduler for deepspline parameters
            try:
                if self.net.deepspline is not None:
                    aux_params_iter = self.net.parameters_deepspline()
            except AttributeError:
                print('Cannot use aux optimizer.')
                raise

            self.aux_optimizer = self.construct_optimizer(
                aux_params_iter, self.optim_names[1], self.params['aux_lr'])

            # scheduler parameters are the same for the main and aux optimizers
            self.aux_scheduler = self.construct_scheduler(self.aux_optimizer)

        if self.params['verbose']:
            self.print_optimization_info()

    @staticmethod
    def construct_optimizer(params_iter, optim_name, lr):
        """
        Construct an optimizer with name 'optim_name'.

        Args:
            params_iter (iter):
                iterator over the parameters to optimize.
            optim_name (str):
                'Adam' or 'SGD' (can be expanded).
            lr (float):
                learning rate.

        Returns:
            optimizer (torch.optim)
        """
        # weight decay is added manually
        if optim_name == 'Adam':
            optimizer = optim.Adam(params_iter, lr=lr)
        elif optim_name == 'SGD':
            optimizer = optim.SGD(params_iter,
                                  lr=lr,
                                  momentum=0.9,
                                  nesterov=True)
        else:
            raise ValueError('Need to provide a valid optimizer type.')

        return optimizer

    def construct_scheduler(self, optimizer):
        """
        Construct a schedule from an optimizer (torch.optim).

        Returns:
            scheduler (torch.optim.lr_scheduler)
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   self.params['milestones'],
                                                   gamma=self.params['gamma'])

        return scheduler

    ##########################################################################
    # TRAIN

    def train(self):
        """ Training loop """

        self.net.train()  # set the network in training mode

        # Load the data
        print('\n==> Loading the data...')
        self.dataloader = DataLoader(self.dataset, **self.params['dataloader'])
        self.trainloader, self.validloader = \
            self.dataloader.get_train_valid_loader()

        self.save_train_info()
        if self.params['verbose']:
            self.print_train_info()

        # Set custom training and validation log steps

        if self.params['log_step'] is None:  # default
            # log at every epoch
            self.params['log_step'] = self.num_train_batches

        if self.params['valid_log_step'] is None:  # default
            # validation done halfway and at the end of training
            self.params['valid_log_step'] = int(
                self.num_train_batches * self.params['num_epochs'] * 1. / 2.)

        elif self.params['valid_log_step'] < 0:
            # validation at every epoch
            self.params['valid_log_step'] = self.num_train_batches

        #####

        # Initialize the losses to log
        # total loss and data fidelity loss
        self.losses_names = ['loss', 'df_loss']

        if self.net.using_deepsplines and self.params['lmbda'] > 0:
            if self.params['lipschitz'] is True:
                self.losses_names.append('bv2_loss')
            else:
                self.losses_names.append('tv2_loss')

        # Set the number of epochs and sparsify_activations flag
        num_epochs = copy.copy(self.params['num_epochs'])
        self.sparsify_activations = False

        if self.params['knot_threshold'] > 0.:
            self.sparsify_activations = True
            # Last added epoch is used to sparsify activations
            num_epochs += 1

        print('\n==> Starting training...')

        for epoch in range(self.start_epoch, num_epochs):

            if epoch == (num_epochs - 1) and self.sparsify_activations is True:
                # sparsify activations and evaluate train accuracy.
                print('\n==> Last epoch: sparsifying activations.')
                self.net.eval()  # set network in evaluation mode
                self.net.sparsify_activations()
                # freeze network to evaluate train accuracy without training
                self.net.freeze_parameters()

            self.train_epoch(epoch)

            if self.dataset.is_user_dataset is True:
                # shuffle training data
                self.trainloader = \
                    self.dataloader.get_shuffled_trainloader_in_memory()

        print('\n==> Training completed.')

        self.log_additional_info()

    def forward_backward_train_batch(self, inputs, labels):
        """
        Forwards a batch and updates the parameter gradients.

        Args:
            inputs, labels (torch.Tensor):
                batch of samples.

        Returns:
            outputs (torch.Tensor)
            losses (list):
                list with the values of the losses corresponding to
                self.losses_names. len(losses) = len(self.losses_names).
        """
        outputs = self.net(inputs)

        data_fidelity = self.criterion(outputs, labels)

        if self.net.training is True:
            data_fidelity.backward()

        losses = [data_fidelity]

        regularization = torch.zeros_like(data_fidelity)
        if self.params['weight_decay'] > 0:
            # weight decay regularization
            wd_regularization = self.params['weight_decay'] / 2 * \
                self.net.l2sqsum_weights_biases()
            regularization = regularization + wd_regularization

        if self.net.using_deepsplines and self.params['lmbda'] > 0:
            # deepspline regularization (TV(2) or BV(2))
            if self.params['lipschitz'] is True:
                ds_regularization = self.params['lmbda'] * self.net.BV2()
            else:
                ds_regularization = self.params['lmbda'] * self.net.TV2()

            losses.append(ds_regularization.clone().detach())
            regularization = regularization + ds_regularization

        if self.net.training is True:
            regularization.backward()

        total_loss = data_fidelity + regularization
        losses.insert(0, total_loss)

        return outputs, losses

    def count_correct(self, outputs, labels):
        """
        Count the number of correct predictions.

        Args:
            outputs (torch.tensor):
                predictions
            labels (torch.tensor)

        Returns:
            correct (int):
                number of correct predictions
        """

        if isinstance(self.criterion, nn.BCELoss):
            predicted = (outputs > 0.5).to(dtype=torch.int64)
            labels = (labels > 0.5).to(dtype=torch.int64)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            _, predicted = outputs.max(1)
        else:
            raise ValueError('Error in criterion (loss type)')

        correct = (predicted == labels).sum().item()

        return correct

    def train_epoch(self, epoch):
        """
        Training for one epoch.

        Args:
            epoch (int)
        """
        print(f'\nEpoch: {epoch}\n')

        running_losses = [0.0] * len(self.losses_names)

        # for computing train accuracy
        correct = 0  # number of correct predictions
        total = 0  # total number of predictions

        for batch_idx, (inputs, labels) in enumerate(self.trainloader):

            if self.net.training is True:
                self.optimizer_zero_grad()

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, losses = self.forward_backward_train_batch(inputs, labels)

            if self.net.training is True:
                self.optimizer_step()

            running_losses = update_running_losses(running_losses, losses)

            correct += self.count_correct(outputs, labels)
            total += labels.size(0)

            if batch_idx % self.params['log_step'] == (
                    self.params['log_step'] - 1):
                # train log step
                train_acc = 100.0 * correct / total
                losses_dict = {
                    key: (value / self.params['log_step'])
                    for (key, value) in zip(self.losses_names, running_losses)
                }
                self.train_log_step(epoch, batch_idx, train_acc, losses_dict)
                # reset values
                running_losses = [0.0] * len(self.losses_names)
                correct, total = 0, 0

            if self.global_step % self.params['valid_log_step'] == (
                    self.params['valid_log_step'] - 1):
                # validation log step
                self.validation_step(epoch)
                self.net.train()

            self.global_step += 1

        if self.net.training is True:
            self.scheduler_step(epoch)

    def scheduler_step(self, epoch):
        """
        Does one step for learning rate schedulers.

        Args:
            epoch (int)
        """
        if self.main_scheduler is not None:
            self.main_scheduler.step()
            if self.params['verbose']:
                main_lr = [
                    group['lr'] for group in self.main_optimizer.param_groups
                ]
                print('main scheduler: epoch - '
                      f'{self.main_scheduler.last_epoch}; '
                      f'learning rate - {main_lr}')

        if self.aux_scheduler is not None:
            self.aux_scheduler.step()
            if self.params['verbose']:
                aux_lr = [
                    group['lr'] for group in self.aux_optimizer.param_groups
                ]
                print('aux scheduler: epoch - '
                      f'{self.aux_scheduler.last_epoch}; '
                      f'learning rate - {aux_lr}')

    def optimizer_zero_grad(self):
        """ Sets parameter gradients to zero """

        self.main_optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()

    def optimizer_step(self):
        """ Updates parameters """

        self.main_optimizer.step()
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()

    def validation_step(self, epoch):
        """
        Does one validation step. Saves results on checkpoint.

        Args:
            epoch (int)
        """
        self.net.eval()

        valid_running_loss = 0.

        # for computing validation accuracy
        correct = 0  # number of correct predictions
        total = 0  # total number of validation samples

        if self.dataset.get_test_imgs:
            plot_dict = self.dataset.init_plot_dict()

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(self.validloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                data_fidelity = self.test_criterion(outputs, labels)
                valid_running_loss += data_fidelity.item()

                correct += self.count_correct(outputs, labels)
                total += labels.size(0)

                if self.dataset.get_test_imgs:
                    self.dataset.add_to_plot_dict(
                        plot_dict, (inputs.cpu(), outputs.cpu()))

        valid_acc = 100.0 * correct / total

        # only add data fidelity loss
        losses_dict = {self.losses_names[1]: (valid_running_loss / total)}

        self.valid_log_step(epoch, valid_acc, losses_dict)
        self.ckpt_log_step(epoch, valid_acc)  # save checkpoint

        if self.dataset.get_test_imgs:
            inputs, outputs = self.dataset.concatenate_plot_dict(plot_dict)
            self.dataset.plot_test_imgs(inputs, outputs)

    ##########################################################################
    # TEST

    def test(self):
        """ Test model """
        self.net.eval()

        print('\n==> Loading the data...')
        self.dataloader = DataLoader(self.dataset, **self.params['dataloader'])
        self.testloader = self.dataloader.get_test_loader()

        if self.params['verbose']:
            self.print_test_info()

        print('\n==> Starting testing...')
        self.forward_test()
        print('\n==> Testing completed.')

    def forward_test(self):
        """ Test loop """

        running_loss = 0.  # running test loss
        # for computing test accuracy
        correct = 0  # number of correct predictions
        total = 0  # total number of test samples

        if self.dataset.get_test_imgs:
            plot_dict = self.dataset.init_plot_dict()

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(self.testloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                running_loss += self.test_criterion(outputs, labels)

                correct += self.count_correct(outputs, labels)
                total += labels.size(0)

                if self.dataset.get_test_imgs:
                    self.dataset.add_to_plot_dict(
                        plot_dict, (inputs.cpu(), outputs.cpu()))

        test_acc = 100.0 * correct / total
        test_loss = running_loss / total
        self.update_json('test_acc', test_acc)
        self.update_json('test_loss', test_loss)

        print('\n=> Test acc  : {:7.3f}%'.format(test_acc))
        print('\n=> Test loss : {:7.3f}'.format(test_loss))

        if self.dataset.get_test_imgs:
            inputs, outputs = self.dataset.concatenate_plot_dict(plot_dict)
            self.dataset.plot_test_imgs(inputs, outputs)
