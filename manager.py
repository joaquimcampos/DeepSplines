# References :
# https://github.com/kuangliu/pytorch-cifar
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from dataloader import DataLoader
from project import Project
from ds_utils import update_running_losses

from models import *
from models.basemodel import MultiResScheduler
from datasets import init_dataset


##################################################################################################

class MyDataParallel(nn.DataParallel):
    " Class for multiple GPU usage "
    def __getattr__(self, name):
        return getattr(self.module, name)

##################################################################################################
#### MANAGER

class Manager(Project):
    """ """

    def __init__(self, params, user_params):
        """ """

        super().__init__(params, user_params)

        loading_success = self.restore_ckpt_params()
        self.init_device()
        self.init_log()

        if self.params['verbose']:
            print('\n==> Parameters info: ', self.params, sep='\n')

        self.init_dataset()
        if self.training and self.params['multires_milestones'] is not None:
            self.init_multires(loading_success)

        self.net = self.build_model(self.params, self.device)

        if self.training:
            self.set_optimization()

        if loading_success is True:
            self.restore_model()

        self.init_json()

        # During testing, average the loss only at the end to get accurate value
        # of the loss per sample. If using reduction='mean', when
        # nb_test_samples % batch_size != 0 we can only average the loss per batch
        # (as done in training for printing the losses) but not per sample.
        if self.params['net'].startswith('twoDnet'):
            self.criterion = nn.BCELoss(reduction='mean')
            self.test_criterion = nn.BCELoss(reduction='sum')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.test_criterion = nn.CrossEntropyLoss(reduction='sum')

        self.criterion.to(self.device)
        self.test_criterion.to(self.device)

        # print(self.net) # for printing network architecture



    def init_dataset(self):
        """ """
        self.dataset = init_dataset(**self.params['dataset'])
        self.params['model']['num_classes'] = self.dataset.num_classes



    @staticmethod
    def build_model(params, device='cuda:0'):
        """ """
        print('\n==> Building model..')

        models_dict = { 'twoDnet_onehidden' : TwoDNet_OneHidden,
                        'twoDnet_twohidden' : TwoDNet_TwoHidden,
                        'simplenet'         : SimpleNet,
                        'simplestnet'       : SimplestNet,
                        'resnet20'          : ResNet20,
                        'resnet32'          : ResNet32 }

        assert params['net'] in models_dict.keys(), 'network not found: please add net to models_dict.'
        net = models_dict[params['net']](**params['model'], device=device)

        net = net.to(device)
        if device == 'cuda':
            # net = MyDataParallel(net) # for multiple GPU usage
            cudnn.benchmark = True

        print('[Network] Total number of parameters : {}'.format(net.num_params))

        return net



    def set_optimization(self):
        """ """
        self.optim_names = self.params['optimizer']

        self.multires_scheduler = None
        if self.params['multires_milestones'] is not None:
            self.multires_scheduler = MultiResScheduler(self.params['multires_milestones'])
            if len(self.optim_names) == 1:
                # use aux optimizer with MultiResScheduler
                self.optim_names.append(self.optim_names[0])

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
                                                    self.optim_names[0], 'main')
        self.main_scheduler = self.construct_scheduler(self.main_optimizer)

        self.aux_optimizer, self.aux_scheduler = None, None

        if len(self.optim_names) == 2:
            # aux optimizer/scheduler for deepspline parameters
            try:
                aux_params_iter = self.net.parameters_deepspline()
            except AttributeError:
                print('Cannot use aux optimizer.')
                raise

            self.aux_optimizer = self.construct_optimizer(aux_params_iter,
                                                    self.optim_names[1], 'aux')
            # scheduler parameters are the same for the main and aux optimizers
            self.aux_scheduler = self.construct_scheduler(self.aux_optimizer)

        if self.params['verbose']:
            self.print_optimization_info()



    def construct_optimizer(self, params_iter, optim_name, mode='main'):
        """ """
        lr = self.params['lr'] if mode == 'main' else self.params['aux_lr']

        # weight decay is added manually
        if optim_name == 'Adam':
            optimizer = optim.Adam(params_iter, lr=lr)
        elif optim_name == 'SGD':
            optimizer = optim.SGD(params_iter, lr=lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError('Need to provide a valid optimizer type')

        return optimizer



    def construct_scheduler(self, optimizer):
        """ """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.params['milestones'],
                                                    gamma=self.params['gamma'])

        return scheduler



##################################################################################################
#### TRAIN

    def train(self):
        """ """
        self.net.train()

        print('\n==> Preparing data..')
        self.dataloader = DataLoader(self.dataset, mode='train', **self.params['dataloader'])
        self.trainloader, self.validloader = self.dataloader.get_train_valid_loader()

        self.save_train_info()

        self.net.init_hyperparams()

        self.losses_names = ['loss', 'df_loss'] # total loss and data fidelity loss
        if self.net.tv_bv_regularization is True:
            self.losses_names.append('tv_bv_loss')

        print('\n\n==>Starting training...')

        for epoch in range(self.start_epoch, self.params['num_epochs']):

            if epoch == (self.params['num_epochs']-1) and self.params['sparsify_activations']:
                print('\nLast epoch: freezing network for sparsifying the activations '
                    'and evaluating training accuracy.')
                self.net.eval() # set network in evaluation mode
                self.net.sparsify_activations()
                self.net.freeze_parameters()

            # optional string to print at each epoch given in params by external script to main_prog()
            if self.params['combination_str'] is not None:
                print(self.params['combination_str'])

            self.train_epoch(epoch)

            if self.dataset.is_user_dataset is True:
                # shuffle training data
                self.trainloader = self.dataloader.get_shuffled_trainloader_in_memory()


        print('\n==> Finished training.')

        self.log_additional_info()



    def forward_backward_train_batch(self, inputs, labels):
        """ Forwards a batch and updates the parameter gradients.
        """
        outputs = self.net(inputs)

        data_fidelity = self.criterion(outputs, labels)

        if self.net.training is True:
            data_fidelity.backward()
            self.net.extra_data_grad_ops()

        losses = [data_fidelity]

        regularization = torch.zeros_like(data_fidelity)
        if self.net.weight_decay_regularization is True:
            # the regularization weight is multiplied inside weight_decay()
            regularization = regularization + self.net.weight_decay()

        if self.net.tv_bv_regularization is True:
            # the regularization weight is multiplied inside TV_BV()
            tv_bv, tv_bv_unweighted = self.net.TV_BV()
            regularization = regularization + tv_bv
            losses.append(tv_bv_unweighted)

        if self.net.training is True:
            regularization.backward()

        total_loss = data_fidelity + regularization
        losses.insert(0, total_loss)

        return outputs, losses



    def count_correct(self, outputs, labels):
        """ count correct predictions """

        if isinstance(self.criterion, nn.BCELoss):
            predicted = (outputs > 0.5).to(dtype=torch.int64)
            labels = (labels  > 0.5).to(dtype=torch.int64)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            _, predicted = outputs.max(1)
        else:
            raise ValueError('Error in criterion (loss type)')

        correct = (predicted == labels).sum().item()

        return correct



    def train_epoch(self, epoch):
        """ """

        print(f'\nEpoch: {epoch}\n')

        running_losses = [0.0] * len(self.losses_names)
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.trainloader):

            if self.net.training is True:
                self.optimizer_zero_grad()

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, losses = self.forward_backward_train_batch(inputs, labels)

            if self.net.training is True:
                if self.params['clip_grad'] is True:
                    torch.nn.utils.clip_grad_value_(self.net.parameters_deepspline(), 0.1)
                self.optimizer_step()

            running_losses = update_running_losses(running_losses, losses)

            correct += self.count_correct(outputs, labels)
            total += labels.size(0)

            if batch_idx % self.params['log_step'] == (self.params['log_step'] - 1):
                train_acc = 100.0 * correct / total
                losses_dict = {key: (value / self.params['log_step']) for (key, value) in zip(self.losses_names, running_losses)}
                self.train_log_step(epoch, batch_idx, train_acc, losses_dict)
                running_losses = [0.0] * len(self.losses_names) # reset running losses
                correct, total = 0, 0

            if self.global_step % self.params['valid_log_step'] == (self.params['valid_log_step'] - 1):
                self.validation_step(epoch)
                self.net.train()

            self.global_step += 1

        if self.net.training is True:
            self.scheduler_step(epoch)



    def scheduler_step(self, epoch):
        """ """
        if self.main_scheduler is not None:
            self.main_scheduler.step()
            if self.params['verbose']:
                main_lr = [group['lr'] for group in self.main_optimizer.param_groups]
                print(f'main scheduler: epoch - {self.main_scheduler.last_epoch}; '
                    f'learning rate - {main_lr}')

        if self.aux_scheduler is not None:
            self.aux_scheduler.step()
            if self.params['verbose']:
                aux_lr = [group['lr'] for group in self.aux_optimizer.param_groups]
                print(f'aux scheduler: epoch - {self.aux_scheduler.last_epoch}; '
                    f'learning rate - {aux_lr}')

        if self.multires_scheduler is not None:
            self.multires_scheduler_step(epoch)
            if self.params['verbose']:
                module = next(self.net.modules_deepspline())
                assert isinstance(module.size, int)
                self.params['model']['spline_size'] = module.size
                self.params['model']['spline_grid'] = module.grid[0].item()
                print('multires scheduler: number of spline coefficients - '
                    f'{self.params["model"]["spline_size"]}; '
                    f'grid size - {self.params["model"]["spline_grid"]}')



    def multires_scheduler_step(self, epoch):
        """ """
        assert self.net.deepspline is not None
        step_done = self.multires_scheduler.step(self.net.modules_deepspline(), epoch)

        if step_done is True:
            # reset aux optimization
            aux_lr = [group['lr'] for group in self.aux_optimizer.param_groups]
            last_epoch = self.aux_scheduler.last_epoch
            aux_params_iter = self.net.parameters_deepspline()
            self.aux_optimizer = self.construct_optimizer(aux_params_iter,
                                                    self.optim_names[1], 'aux')
            self.aux_scheduler = self.construct_scheduler(self.aux_optimizer)
            for i, g in enumerate(self.aux_optimizer.param_groups):
                g['lr'] = aux_lr[i]
            self.aux_scheduler.last_epoch = last_epoch



    def optimizer_zero_grad(self):
        """ """
        self.main_optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()
        self.net.extra_zero_grad_ops()



    def optimizer_step(self):
        """ """
        self.main_optimizer.step()
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()
        self.net.extra_parameter_update_ops()



    def validation_step(self, epoch):
        """ """
        self.net.eval()

        valid_running_loss = 0.
        valid_correct, valid_total = 0, 0

        if self.dataset.get_plot:
            plot_dict = self.dataset.init_plot_dict()

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(self.validloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                data_fidelity = self.test_criterion(outputs, labels)
                valid_running_loss += data_fidelity.item()

                valid_correct += self.count_correct(outputs, labels)
                valid_total += labels.size(0)

                if self.dataset.get_plot:
                    self.dataset.add_to_plot_dict(plot_dict, (inputs.cpu(), outputs.cpu()))


        valid_acc = 100.0 * valid_correct / valid_total

        # only add data fidelity loss
        losses_dict = {self.losses_names[1] : (valid_running_loss / valid_total)}

        self.valid_log_step(epoch, valid_acc, losses_dict)
        self.ckpt_log_step(epoch, valid_acc) # save checkpoint

        if self.dataset.get_plot:
            inputs, outputs = self.dataset.concatenate_plot_dict(plot_dict)
            self.dataset.plot_test_imgs(inputs, outputs)



##################################################################################################
#### TEST

    def test(self):
        """ """
        self.net.eval()

        print('\n==> Preparing data..')
        self.dataloader = DataLoader(self.dataset, mode='test', **self.params['dataloader'])
        self.testloader = self.dataloader.get_test_loader()

        self.save_test_info()

        self.forward_test()
        print('\nFinished testing.')



    def forward_test(self):
        """ forwards test samples and calculates the test accuracy """

        running_loss = 0.
        correct = 0
        total = 0

        if self.dataset.get_plot:
            plot_dict = self.dataset.init_plot_dict()

        if self.params['sparsify_activations']:
            self.net.sparsify_activations()


        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(self.testloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                running_loss += self.test_criterion(outputs, labels)

                correct += self.count_correct(outputs, labels)
                total += labels.size(0)

                if self.dataset.get_plot:
                    self.dataset.add_to_plot_dict(plot_dict, (inputs.cpu(), outputs.cpu()))


        test_acc  = 100.0 * correct / total
        test_loss = running_loss / total
        self.update_json('test_acc', test_acc)
        self.update_json('test_loss', test_loss)

        print('\n=> Test acc  : {:7.3f}%'.format(test_acc))
        print('\n=> Test loss : {:7.3f}'.format(test_loss))

        if self.dataset.get_plot:
            inputs, outputs = self.dataset.concatenate_plot_dict(plot_dict)
            self.dataset.plot_test_imgs(inputs, outputs)
