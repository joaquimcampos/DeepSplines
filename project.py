import torch
import os
import glob
import math
import sys
import collections
import itertools

from ds_utils import size_str, dict_recursive_merge, flatten_structure
from ds_utils import json_load, json_dump


class Project():

    train_results_json_filename = 'train_results.json'
    test_results_json_filename = 'test_results.json'
    train_sorting_key = 'latest_valid_acc'
    test_sorting_key = 'test_acc'


    def __init__(self, params, user_params):

        self.params = params
        self.user_params = user_params
        self.training = (self.params["mode"]=='train')



    def init_log(self):
        """ Create Log directory for training the model as :
            self.params["log_dir"]/self.params["model_name"]/
        """
        self.log_dir_model = os.path.join(self.params["log_dir"],
                                            self.params["model_name"])
        if not os.path.isdir(self.log_dir_model):
            os.makedirs(self.log_dir_model)



    def init_device(self):
        """ """
        if self.params['device'] == 'cuda:0':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print('\nUsing GPU.')
            else:
                self.device = 'cpu'
                print('\nCUDA not available. Using CPU.')
        else:
            self.device = 'cpu'
            print('\nUsing CPU.')



    @property
    def results_json_filename(self):
        if self.training is True:
            return self.train_results_json_filename
        else:
            return self.test_results_json_filename


    @property
    def sorting_key(self):
        """ Key for sorting models in json file.
        """
        if self.training:
            return self.train_sorting_key
        else:
            return self.test_sorting_key



    def init_json(self):
        """ Init json file for train/test results.
        """
        # initialize/verify json log file
        self.results_json = os.path.join(self.params['log_dir'],
                                        self.results_json_filename)

        if not os.path.isfile(self.results_json):
            results_dict = {}
        else:
            results_dict = json_load(self.results_json)

        if self.params['model_name'] not in results_dict:
            results_dict[self.params['model_name']] = {} # initialize model log

        # add minimal information for sorting models in results_json file
        if self.sorting_key not in results_dict[self.params['model_name']]:
            results_dict[self.params['model_name']][self.sorting_key] = 0.

        json_dump(results_dict, self.results_json)

        comb_list = list(itertools.product(['latest', 'best'], ['train', 'valid'],['acc', 'loss']))
        self.info_list = ['_'.join(k) for k in comb_list] + ['test_acc', 'test_loss']



    def update_json(self, info, value):
        """ Update json file with latest/best validation/test accuracy/loss, if training,
        and with test accuracy otherwise

        Args:
            info: e.g. 'latest_validation_loss'
        """
        assert info in self.info_list, f'{info} should be in {self.info_list}...'

        # save in json
        results_dict = json_load(self.results_json)

        if isinstance(value, dict):
            if info not in self.params["model_name"]:
                results_dict[self.params["model_name"]][info] = {}
            for key, val in value.items():
                results_dict[self.params["model_name"]][info][key] = float('{:.3f}'.format(val))
        else:
            results_dict[self.params["model_name"]][info] = float('{:.3f}'.format(value))

        sorted_acc = sorted(results_dict.items(), key=lambda kv : kv[1][self.sorting_key], reverse=True)
        sorted_results_dict = collections.OrderedDict(sorted_acc)

        json_dump(sorted_results_dict, self.results_json)



    def restore_ckpt_params(self):
        """ Restore the parameters from a previously saved checkpoint
        (either provided via --ckpt_filename or saved in log_dir/model_name)

        Returns:
            True if a checkpoint was successfully loaded and False otherwise.
        """
        self.best_train_acc = 0.
        self.best_valid_acc = 0.

        if self.training:
            self.start_epoch, self.global_step = 0, 0

        if self.params["ckpt_filename"] is not None and self.params["ckpt_filename"] != '':
            try:
                self.load_merge_params(self.params["ckpt_filename"])

            except FileNotFoundError:
                print('\nCheckpoint file not found... Unable to restore model.\n')
                raise

            except:
                print('\nUnknown error in restoring model.')
                raise

            print('\nSuccessfully loaded ckpt ' + self.params["ckpt_filename"])
            return True

        elif self.params["resume"] is True:

            log_dir_model = os.path.join(self.params["log_dir"], self.params["model_name"])
            if self.params["resume_from_best"] is True:
                regexp_ckpt = os.path.join(log_dir_model, '*_best_valid_acc.pth')
            else:
                regexp_ckpt = os.path.join(log_dir_model, '*_net_*.pth')

            files = glob.glob(regexp_ckpt)
            files.sort(key=os.path.getmtime) # sort by time from oldest to newest

            if len(files) > 0:
                print('\nRestoring model from {}.'.format(files[-1]))
                self.load_merge_params(files[-1]) # restore from most recent file
                return True
            else:
                print('\nNo model saved to resume training. Starting from scratch.')
                return False

        else:
            print('\nStarting from scratch.')
            return False



    def load_merge_params(self, ckpt_filename):
        """ Load and merge the parameters from ckpt_filename into self.params

        The parameters introduced by the user (via command-line arguments)
        override the corresponding saved parameters. The ones not specified
        by the user, are loaded from the checkpoint.
        """
        torch.load(ckpt_filename, map_location = lambda storage, loc: storage)
        ckpt = self.get_loaded_ckpt(ckpt_filename)
        self.loaded_ckpt = ckpt # save loaded_ckpt for restore_model

        saved_params = ckpt['params']
        self.params = dict_recursive_merge(self.params, saved_params) # merge w/ saved params
        self.params = dict_recursive_merge(self.params, self.user_params)  # merge w/ user params (precedence over saved)



    def restore_model(self):
        """ """
        self.load_model(self.loaded_ckpt)

        if self.training and self.start_epoch == self.params["num_epochs"]:
            print('\nTraining in this checkpoint is already completed. '
                    'Please increase the number of epochs.')
            sys.exit()



    def load_model(self, ckpt):
        """ """
        # Load checkpoint.
        print('\n==> Resuming from checkpoint...')

        self.net.load_state_dict(ckpt['model_state'], strict=(self.training is True))
        self.best_train_acc = ckpt['best_train_acc']
        self.best_valid_acc = ckpt['best_valid_acc']

        if self.training:
            self.start_epoch = ckpt['num_epochs_finished']
            self.global_step = ckpt['global_step']
            self.main_optimizer.load_state_dict(ckpt['main_optimizer_state'])

            if ckpt['aux_optimizer_state'] is not None:
                self.aux_optimizer.load_state_dict(ckpt['aux_optimizer_state'])

            if 'main_scheduler_state' in ckpt:
                self.main_scheduler.load_state_dict(ckpt['main_scheduler_state'])
                if ckpt['aux_scheduler_state'] is not None:
                    self.aux_scheduler.load_state_dict(ckpt['aux_scheduler_state'])

        return



    @staticmethod
    def get_loaded_ckpt(ckpt_filename):
        """ Returns a loaded checkpoint from ckpt_filename, if it exists.
        """
        try:
            # TODO: Check if model is always loaded on cpu. Use net.to(device) after.
            ckpt = torch.load(ckpt_filename, map_location = lambda storage, loc: storage)

        except FileNotFoundError:
            print('\nCheckpoint file not found... Unable to load checkpoint.\n')
            raise
        except:
            print('\nUnknown error in loading checkpoint parameters.')
            raise

        return ckpt



    @classmethod
    def load_ckpt_params(cls, ckpt_filename, flatten=True):
        """ Returns the parameters saved in a checkpoint.

        Args:
            flatten - whether to flatten the structure of the parameters dictionary
            into a single level (see structure in struct_default_values.py)
        """
        ckpt = cls.get_loaded_ckpt(ckpt_filename)
        params = ckpt['params']

        if flatten is True:
            params = flatten_structure(params)

        return ckpt, params



    @staticmethod
    def get_ckpt_from_log_dir_model(log_dir_model):
        """ Get last ckpt from log_dir_model (log_dir/model_name)
        """
        regexp_ckpt = os.path.join(log_dir_model, '*_net_*.pth')

        files = glob.glob(regexp_ckpt)
        files.sort(key=os.path.getmtime) # sort by time from oldest to newest

        if len(files) > 0:
            ckpt_filename = files[-1]
            print(f'Restoring {ckpt_filename}')
            return ckpt_filename
        else:
            print(f'No ckpt found in {log_dir_model}...')
            return None



    @classmethod
    def load_results_dict(cls, log_dir, mode='train'):
        """ Load results dictionary from json file corresponding to the
        train or test results in log_dir.
        """
        assert mode in ['train', 'test'], 'mode should be "train" or "test"...'
        if mode == 'train':
            results_json_filename = cls.train_results_json_filename
        else:
            results_json_filename = cls.test_results_json_filename

        results_json = os.path.join(log_dir, results_json_filename)
        results_dict = json_load(results_json)

        return results_dict



    @classmethod
    def dump_results_dict(cls, results_dict, log_dir, mode='train'):
        """ Dump results dictionary in json file corresponding to the
        train or test results in log_dir.
        """
        assert mode in ['train', 'test'], 'mode should be "train" or "test"...'
        if mode == 'train':
            results_json_filename = cls.train_results_json_filename
        else:
            results_json_filename = cls.test_results_json_filename

        results_json = os.path.join(log_dir, results_json_filename)
        json_dump(results_dict, results_json)



    @classmethod
    def get_best_model(cls, log_dir, mode='train'):
        """ Get the name and checkpoint of the best model (best validation/test)
        from the train/test results (saved in log_dir/[mode]_results.json).
        """
        results_dict = cls.load_results_dict(log_dir, mode)

        # models are ordered by validation accuracy; choose first one.
        best_model_name = next(iter(results_dict))
        log_dir_best_model = os.path.join(log_dir, best_model_name)
        ckpt_filename = cls.get_ckpt_from_log_dir_model(log_dir_best_model)

        return best_model_name, ckpt_filename



    def train_log_step(self, epoch, batch_idx, train_acc, losses_dict):
        """
        Args:
            losses_dict - a dictionary {loss name : loss value}
        """
        print('[{:3d}, {:6d} / {:6d}] '.format(epoch + 1, batch_idx + 1, self.num_train_batches), end='')
        for key, value in losses_dict.items():
            print('{}: {:7.3f} | '.format(key, value), end='')

        print('train acc: {:7.3f}%'.format(train_acc))

        self.update_json('latest_train_loss', losses_dict)
        self.update_json('latest_train_acc', train_acc)

        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc
            self.update_json('best_train_acc', train_acc)



    def valid_log_step(self, epoch, valid_acc, losses_dict):
        """
        Args:
            losses_dict - a dictionary {loss name : loss value}
        """
        print('\nvalidation_step : ', end='')
        for key, value in losses_dict.items():
            print('{}: {:7.3f} | '.format(key, value), end='')

        print('valid acc: {:7.3f}%'.format(valid_acc), '\n')

        self.update_json('latest_valid_loss', losses_dict)
        self.update_json('latest_valid_acc', valid_acc)

        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.update_json('best_valid_acc', valid_acc)



    def ckpt_log_step(self, epoch, valid_acc):
        """ """
        base_ckpt_filename = os.path.join(self.log_dir_model, self.params["model_name"] + '_net_{:04d}'.format(epoch+1))
        regexp_ckpt = os.path.join(self.log_dir_model, "*_net_*.pth")
        regexp_best_valid_acc_ckpt = os.path.join(self.log_dir_model, "*_best_valid_acc.pth")

        # save checkpoint as *_net_{epoch+1}.pth
        ckpt_filename = base_ckpt_filename + '.pth'

        files = list(set(glob.glob(regexp_ckpt)) - set(glob.glob(regexp_best_valid_acc_ckpt))) # remove best_valid_acc ckpt from files
        files.sort(key=os.path.getmtime, reverse=True) # sort from newest to oldest

        if (not self.params["ckpt_nmax_files"] < 0) and (len(files) >= self.params["ckpt_nmax_files"]):
            assert len(files) == (self.params["ckpt_nmax_files"]), 'There are more than (ckpt_nmax_files+1) *_net_*.pth checkpoints.'
            filename = files[-1]
            os.remove(filename)

        self.save_network(ckpt_filename, epoch, valid_acc)

        if valid_acc == self.best_valid_acc:
            # if valid_acc = best_valid_acc, also save checkpoint as *_net_{global_step}_best_valid_acc.pth
            # and delete previous best_valid_acc checkpoint
            best_valid_acc_ckpt_filename = base_ckpt_filename + '_best_valid_acc.pth'
            files = glob.glob(regexp_best_valid_acc_ckpt)

            if len(files) > 0:
                assert len(files) == 1, 'There is more than one *_best_valid_acc.pth checkpoint.'
                os.remove(files[0])

            self.save_network(best_valid_acc_ckpt_filename, epoch, valid_acc)


        return



    def save_network(self, ckpt_filename, epoch, valid_acc):
        """

        ckpt_filename: where model is saved
        valid_acc:  accuracy
        epofch: current epoch
        """
        state = {
            'model_state'          : self.net.state_dict(),
            'main_optimizer_state' : self.main_optimizer.state_dict(),
            'main_scheduler_state' : self.main_scheduler.state_dict(),
            'params'               : self.params,
            'best_train_acc'       : self.best_train_acc,
            'best_valid_acc'       : self.best_valid_acc,
            'valid_acc'            : valid_acc,
            'num_epochs_finished'  : epoch + 1,
            'global_step'          : self.global_step
        }

        if self.aux_optimizer is not None:
            state['aux_optimizer_state'] = self.aux_optimizer.state_dict()
            state['aux_scheduler_state'] = self.aux_scheduler.state_dict()
        else:
            state['aux_optimizer_state'] = None
            state['aux_scheduler_state'] = None

        torch.save(state, ckpt_filename)

        return



    def save_train_info(self):
        """ """
        assert (self.trainloader is not None) and (self.validloader is not None)

        if self.dataset.is_user_dataset is True:
            num_train_samples = sum(inputs.size(0) for inputs,_ in self.trainloader)
            num_valid_samples = sum(inputs.size(0) for inputs,_ in self.validloader)
        else:
            num_train_samples = len(self.trainloader.sampler)
            num_valid_samples = len(self.validloader.sampler)

        self.num_train_batches = math.ceil(num_train_samples / self.dataloader.batch_size)
        num_valid_batches = math.ceil(num_valid_samples / self.dataloader.batch_size)

        if self.params['verbose']:
            if self.dataset.is_user_dataset is True:
                sample_data, sample_target = self.trainloader[0]
            else:
                # dataloader iterator to get next sample
                dataiter = iter(self.trainloader)
                sample_data, sample_target = dataiter.next()

            print('\n==> Training info:')
            print(f'batch (data, target) size : ({size_str(sample_data)}, {size_str(sample_target)})')
            print(f'no. of (train, valid) samples : ({num_train_samples}, {num_valid_samples})')
            print(f'no. of (train, valid) batches : ({self.num_train_batches}, {num_valid_batches})')



    def save_test_info(self):
        """ """
        assert (self.testloader is not None)

        if self.dataset.is_user_dataset is True:
            num_test_samples = sum(inputs.size(0) for inputs,_ in self.testloader)
        else:
            num_test_samples = len(self.testloader.dataset)

        num_test_batches = math.ceil(num_test_samples / self.dataloader.batch_size)

        if self.params['verbose']:
            if self.dataset.is_user_dataset is True:
                sample_data, sample_target = self.testloader[0]
            else:
                # dataloader iterator to get next sample
                dataiter = iter(self.testloader)
                sample_data, sample_target = dataiter.next()

            print('\n==> Test info:')
            print(f'batch (data, target) size : ({size_str(sample_data)}, {size_str(sample_target)})')
            print(f'no. of test (samples, batches) : ({num_test_samples}, {num_test_batches})')



    def print_optimization_info(self):
        """ """
        print('\n==> Optimizer info:')
        print('--Main Optimizer:')
        print(self.main_optimizer)
        if self.aux_optimizer is not None:
            print('--Aux Optimizer  :')
            print(self.aux_optimizer)

        # scheduler
        scheduler_list = [self.main_scheduler, self.aux_scheduler]
        scheduler_name_list = ['Main', 'Aux']
        for scheduler, aux_str in zip(scheduler_list, scheduler_name_list):
            if scheduler is not None:
                print('--' + aux_str + ' Scheduler : ')
                print(f'class - {type(scheduler).__name__}; '
                    f'milestones - {scheduler.milestones}; '
                    f'gamma - {scheduler.gamma}.')



    def log_additional_info(self):
        """ Log additional information to self.results_json
        """
        if not self.params['additional_info']: # empty list
            return

        results_dict = json_load(self.results_json)

        if 'sparsity' in self.params['additional_info']:
            results_dict[self.params['model_name']]['sparsity'] = \
                                        '{:d}'.format(self.net.compute_sparsity())

        if 'lipschitz_bound' in self.params['additional_info']:
            results_dict[self.params['model_name']]['lipschitz_bound'] = \
                                        '{:.3f}'.format(self.net.lipschitz_bound())

        json_dump(results_dict, self.results_json)
