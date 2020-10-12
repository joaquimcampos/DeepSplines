################################
# deepspline project utilities
################################

import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import math
import json
import numpy as np
import copy
import collections


class ArgCheck():
    """ Class for input argument verification """

    @staticmethod
    def p_int(value):
        """ Check if int value got from argparse is positive
            and raise error if not.
        """
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')
        return ivalue


    @staticmethod
    def p_odd_int(value):
        """ Check if int value got from argparse is positive
            and raise error if not.
        """
        ivalue = int(value)
        if (ivalue <= 0) or ((ivalue + 1) % 2 != 0) :
            raise argparse.ArgumentTypeError(f'{value} is an invalid positive odd int value')
        return ivalue


    @staticmethod
    def nn_int(value):
        """ Check if int value got from argparse is non-negative
            and raise error if not.
        """
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f'{value} is an invalid non-negative int value')
        return ivalue



    @staticmethod
    def p_float(value):
        """ Check if float value got from argparse is positive
            and raise error if not.
        """
        ivalue = float(value)
        if ivalue <= 0:
             raise argparse.ArgumentTypeError(f'{value} is an invalid positive float value')
        return ivalue



    @staticmethod
    def n_float(value):
        """ Check if float value got from argparse is negative
            and raise error if not.
        """
        ivalue = float(value)
        if ivalue >= 0:
             raise argparse.ArgumentTypeError(f'{value} is an invalid negative float value')
        return ivalue



    @staticmethod
    def frac_float(value):
        """ Check if float value got from argparse is >= 0 and <= 1
            and raise error if not.
        """
        ivalue = float(value)
        if ivalue < 0 or ivalue > 1:
             raise argparse.ArgumentTypeError(f'{value} is an invalid fraction float value (should be in [0, 1])')
        return ivalue


    @staticmethod
    def nn_float(value):
        """ Check if float value got from argparse is non-negative
            and raise error if not.
        """
        ivalue = float(value)
        if not np.allclose(np.clip(ivalue, -1.0, 0.0), 0.0):
            raise argparse.ArgumentTypeError(f'{value} is an invalid non-negative float value')
        return ivalue



def size_str(input):
    """ Returns a string with the size of the input pytorch tensor
    """
    out_str = '[' + ', '.join(str(i) for i in input.size()) + ']'
    return out_str



def update_running_losses(running_losses, losses):
    """ Update the running_losses with the newly calculated losses

    len(running_losses) = len(losses)
    """
    for i, loss in enumerate(losses):
        running_losses[i] += loss.item()

    return running_losses



def denormalize(img_tensor, mean_tuple, std_tuple):
    """ Denormalize input images using (mean_tuple, std_tuple)
    """
    mean = Tensor([mean_tuple]).view(1, -1, 1, 1)
    std  = Tensor([std_tuple]).view(1, -1, 1, 1)

    img_tensor = img_tensor * std + mean

    return img_tensor



def dict_recursive_merge(params_root, merger_root, base=True):
    """ Recursively merges merger_root into params_root giving precedence
    to the second as in z = {**x, **y}
    """
    if base:
        assert isinstance(params_root, dict)
        assert isinstance(merger_root, dict)
        merger_root = copy.deepcopy(merger_root)

    if merger_root: # non-empty dict
        for key, val in merger_root.items():
            if isinstance(val, dict):
                merger_root[key] = dict_recursive_merge(params_root[key], merger_root[key], base=False)

        merger_root = {**params_root, **merger_root}

    return merger_root



def assign_structure_recursive(assign_root, structure, base=True):
    """ Recursively assigns values to assign_root according to structure
    (see structure variable in default_struct_values.py)
    """
    if base:
        assert isinstance(assign_root, dict)
        assert isinstance(structure, dict)
        assign_root = copy.deepcopy(assign_root)
        global assign_orig
        assign_orig = assign_root # keep the original dict
        global leaves
        leaves = [] # leaves on dict tree levels deeper than base

    if structure: # non-empty dict
        for key, val in structure.items():
            if isinstance(val, dict):
                assign_root[key] = {}
                assign_root[key] = assign_structure_recursive(assign_root[key], structure[key], base=False)
                if len(assign_root[key]) < 1: # do not have empty dictionaries in assign_root
                    del assign_root[key]
            else:
                assert val is None, 'leaf values in structure should be None'
                if key in assign_orig:
                    assign_root[key] = assign_orig[key]
                    if not base:
                        leaves.append(key)

    # delete duplicated leaves in base root if they are not in first level of structure dict
    if base:
        for key in leaves:
            if key not in structure and key in assign_root:
                del assign_root[key]


    return assign_root



def flatten_structure(assign_root, base=True):
    """ Flattens the structure created with assign_structure_recursive()
    """
    if base:
        assert isinstance(assign_root, dict)
        global flattened
        flattened = {}
        assign_root = copy.deepcopy(assign_root)

    for key, val in assign_root.items():
        if isinstance(val, dict):
            flatten_structure(assign_root[key], base=False)
        elif key not in flattened:
            flattened[key] = val

    return flattened



def check_device(*tensors, dev='cpu'):
    """ Check if tensors are in device dev
    """
    for t in tensors:
        assert t.device == torch.device(dev), f'tensor is not in device {dev}'



def spline_grid_from_range(spline_size, range_=2, round_to=1e-6):
    """ Compute spline grid spacing from desired one-side range
    and the number of activation coefficients.

    Args:
        round_to: round grid to this value
    """
    spline_grid = ((range_ / (spline_size//2)) // round_to) * round_to

    return spline_grid



def init_sub_dir(top_dir, sub_dir_name):
    """ Initialize and return sub directory. Create it if it does not exist.
    """
    sub_dir = os.path.join(top_dir, sub_dir_name)
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)

    return sub_dir



def json_load(json_filename):
    """ """
    try:
        with open(json_filename) as jsonfile:
            results_dict = json.load(jsonfile)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise

    return results_dict



def json_dump(results_dict, json_filename):
    """ """
    try:
        with open(json_filename, 'w') as jsonfile:
            json.dump(results_dict, jsonfile, sort_keys=False, indent=4)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise



def isPowerOfTwo(n):
    return (math.ceil(np.log2(n)) == math.floor(np.log2(n)))
