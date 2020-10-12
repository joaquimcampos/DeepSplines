#!/usr/bin/env python3

# References:
# [1] https://github.com/kuangliu/pytorch-cifar
# [2] https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# [3] https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

import os
import torch
import torchvision
from torch.utils.data import SubsetRandomSampler
import numpy as np
import math
from random import randint

from ds_utils import size_str


class DataLoader():
    """ """

    def __init__(self, dataset, mode='train', data_dir='./data', batch_size=64,
                num_workers=4, train=None, **kwargs):
        """

        Args:
            dataset: Dataset class instance (datasets.py)
            mode: 'train' or 'test'
            data_dir: path directory where data is stored
        """
        assert mode in ['train', 'test'], 'mode should be "train" or "test".'

        self.mode = mode
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()

        if self.mode == 'train':
            self.train_params = train



    def get_dataset_dir(self):
        """ Returns self.data_dir/self.dataset/
        """
        return os.path.join(self.data_dir, self.dataset.name)



    def get_loader_in_memory(self, inputs, labels, batch_size=None):
        """ Split the data in batches
        """
        minibatch = self.batch_size if batch_size is None else batch_size
        dataloader = list(zip(inputs.split(minibatch), labels.split(minibatch)))

        return dataloader



    def shuffle_data_in_memory(self, inputs, labels):
        """ Shuffle data when tensors are in memory
        """
        permutation_idx = torch.randperm(inputs.size(0))
        inputs = torch.index_select(inputs, 0, permutation_idx)
        labels = torch.index_select(labels, 0, permutation_idx)

        return inputs, labels



    def get_shuffled_trainloader_in_memory(self):
        """ Get reshufled trainloader when tensors are in memory

        Splits the data in random batches
        """
        train_inputs, train_labels = self.shuffle_data_in_memory(self.train_inputs, self.train_labels)
        trainloader = self.get_loader_in_memory(train_inputs, train_labels)

        return trainloader



    def load_dataset_in_memory(self, mode):
        """ Load dataset saved in memory
        """
        assert mode in ['train', 'valid', 'test']

        dataset_dir = self.get_dataset_dir()
        if not os.path.isdir(dataset_dir):
            raise OSError(f'Directory {dataset_dir} does not exist.')

        loaded_data = torch.load(os.path.join(dataset_dir, mode + '_data.pth'))
        inputs, labels = loaded_data['inputs'], loaded_data['labels']

        if mode == 'train' and self.dataset.get_plot:
            self.dataset.plot_train_imgs(inputs, labels)

        return inputs, labels



    def get_train_valid_loader(self, shuffle=True):
        """ Get the training and validation loaders (iterators) for batch training.

        Returns:
            trainloader: training set iterator
            validloader: validation set iterator
        """
        if self.dataset.is_user_dataset is True:

            self.train_inputs, self.train_labels = self.load_dataset_in_memory('train')
            valid_inputs, valid_labels = self.load_dataset_in_memory('valid')

            trainloader = self.get_shuffled_trainloader_in_memory() # shuffle self.train_inputs/label pairs
            validloader = self.get_loader_in_memory(valid_inputs, valid_labels, batch_size=100)

            return trainloader, validloader


        train_transform, valid_transform = self.dataset.get_train_valid_transforms()

        torchvision_dataset = self.dataset.get_torchvision_dataset()
        train_dataset = torchvision_dataset(self.get_dataset_dir(), train=True,
                                        download=True, transform=train_transform)

        if self.train_params['test_as_valid']:
            # use test dataset for validation
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
            validloader = self.get_test_loader()
        else:
            # sampler train/val split: valid_dataset will be a subset of the training data
            valid_dataset = torchvision_dataset(self.get_dataset_dir(), train=True,
                                            download=True, transform=valid_transform)

            train_indices, valid_indices = self.get_split_indices(train_dataset, shuffle)

            train_sampler = SubsetRandomSampler(train_indices) # sample elements randomly, without replacement from train_indices
            valid_sampler = SubsetRandomSampler(valid_indices)

            # in Dataloader shuffle=False since we already shuffle the train/valid datasets
            # through shuffling the indices and using SubsetRandomSampler
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                sampler=train_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,
                                sampler=valid_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)


        if self.dataset.plot_imgs:
            self.dataset.plot_train_imgs(trainloader)

        return trainloader, validloader



    def get_split_indices(self, train_dataset, shuffle=True):
        """ Get training and validation random split indices.

        The number of validation samples (indices) is a small fraction of the
        total training_dataset. The remaining indices correspond to the training data.
        """
        num_train_samples = len(train_dataset)
        indices = list(range(num_train_samples))

        valid_fraction = 0.1 # fraction of training data to use for validation
        assert (valid_fraction >= 0) and (valid_fraction <= 1), 'valid_fraction should be in the range [0, 1].'
        split = int(np.floor(valid_fraction * num_train_samples))

        if shuffle:
            seed = self.train_params['seed']
            seed = seed if seed >= 0 else 0
            np.random.seed(seed) # random seed so that validation data is always the same throughout training
            np.random.shuffle(indices) # shuffle indices (same as shuffling train and valid set)

        # indices are already shuffled
        train_indices, valid_indices = indices[split:], indices[:split]

        return train_indices, valid_indices



    def get_test_loader(self):
        """ Utility function for loading and returning a multi-process
            test iterator over the dataset (for batch testing).

        Returns:
            testloader: test set iterator.
        """

        if self.dataset.is_user_dataset is True:
            test_inputs, test_labels = self.load_dataset_in_memory('test')
            testloader = self.get_loader_in_memory(test_inputs, test_labels, batch_size=100)

            return testloader

        test_transform = self.dataset.get_test_transform()

        torchvision_dataset = self.dataset.get_torchvision_dataset()
        test_dataset = torchvision_dataset(self.get_dataset_dir(), train=False,
                                        download=True, transform=test_transform)

        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return testloader
