#!/usr/bin/env python3

import argparse
import torch
import numpy as np

from project import Project
from manager import Manager
from datasets import init_dataset
from dataloader import DataLoader

def get_input_statistics(ckpt_filename):

    ckpt = Project.get_loaded_ckpt(ckpt_filename)
    params = ckpt['params']
    acc = ckpt['acc']

    assert params['net'] == 'twoDnet_onehidden'

    # print('\nLoading parameters from checkpoint : ', ckpt_filename, sep='\n')
    # print('\nParameters : ', params, sep='\n')
    # print('accuracy : {:.3f}%'.format(acc))

    net  = Manager.build_model(params, device=params['device'])
    net.load_state_dict(ckpt['model_state'], strict=True)
    net.eval()

    with torch.no_grad():

        dataset = init_dataset(**params['dataset'])
        dataloader = DataLoader(dataset, mode='train', **params['dataloader'])
        inputs, _ = dataloader.load_dataset_in_memory()

        x = net.get_inputs_to_activation(inputs).detach().cpu().numpy()

        x_min = np.max(x, axis=0)
        x_max = np.min(x, axis=0)

        return x_min, x_max



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Get activation input statistics of model from checkpoint file.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ckpt_filename', type=str, help='')
    args = parser.parse_args()

    x_min, x_max = get_input_statistics(args.ckpt_filename)

    print('\n\nx_min :', x_min)
    print('x_max :', x_max)
