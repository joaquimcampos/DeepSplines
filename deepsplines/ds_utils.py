""" Module with project utilities """

import os
import argparse
import copy
import json
from datetime import datetime
import torch
from torch import Tensor
import numpy as np


class ArgCheck():
    """ Class for input argument verification """
    @staticmethod
    def p_int(value):
        """
        Check if int value got from argparse is positive
        and raise error if not.
        """
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid positive int value')
        return ivalue

    @staticmethod
    def p_odd_int(value):
        """
        Check if int value got from argparse is positive
        and raise error if not.
        """
        ivalue = int(value)
        if (ivalue <= 0) or ((ivalue + 1) % 2 != 0):
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid positive odd int value')
        return ivalue

    @staticmethod
    def nn_int(value):
        """
        Check if int value got from argparse is non-negative
        and raise error if not.
        """
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid non-negative int value')
        return ivalue

    @staticmethod
    def p_float(value):
        """
        Check if float value got from argparse is positive
        and raise error if not.
        """
        ivalue = float(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid positive float value')
        return ivalue

    @staticmethod
    def n_float(value):
        """
        Check if float value got from argparse is negative
        and raise error if not.
        """
        ivalue = float(value)
        if ivalue >= 0:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid negative float value')
        return ivalue

    @staticmethod
    def frac_float(value):
        """
        Check if float value got from argparse is >= 0 and <= 1
        and raise error if not.
        """
        ivalue = float(value)
        if ivalue < 0 or ivalue > 1:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid fraction float value '
                '(should be in [0, 1])')
        return ivalue

    @staticmethod
    def nn_float(value):
        """
        Check if float value got from argparse is non-negative
        and raise error if not.
        """
        ivalue = float(value)
        if not np.allclose(np.clip(ivalue, -1.0, 0.0), 0.0):
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid non-negative float value')
        return ivalue


def size_str(input):
    """
    Returns a string with the size of the input tensor

    Args:
        input (torch.Tensor)
    Returns:
        out_str (str)
    """
    out_str = '[' + ', '.join(str(i) for i in input.size()) + ']'
    return out_str


def update_running_losses(running_losses, losses):
    """
    Update the running_losses with the newly calculated losses.

    len(running_losses) = len(losses).

    Args:
        running_losses (list)
        losses (list)
    """
    for i, loss in enumerate(losses):
        running_losses[i] += loss.item()

    return running_losses


def denormalize(img_tensor, mean_tuple, std_tuple):
    """
    Denormalize input images.

    Args:
        img_tensor (torch.Tensor):
            Tensor of size (N, C, H, W)
        mean_tuple (tuple):
            C-tuple with mean for each channel
        std_tuple (tuple):
            C-tuple with standard deviation for each channel
    """
    assert len(img_tensor.size()) == 4, f'{len(img_tensor.size())} != 4.'
    assert img_tensor.size(1) == len(mean_tuple), \
        f'{img_tensor.size(1)} != {len(mean_tuple)}.'
    assert img_tensor.size(1) == len(std_tuple), \
        f'{img_tensor.size(1)} != {len(std_tuple)}.'

    mean = Tensor([mean_tuple]).view(1, -1, 1, 1)
    std = Tensor([std_tuple]).view(1, -1, 1, 1)

    img_tensor = img_tensor * std + mean

    return img_tensor


def dict_recursive_merge(params_root, merger_root, base=True):
    """
    Recursively merges merger_root into params_root giving precedence
    to the second dictionary as in z = {**x, **y}

    Args:
        params_root (dict)
        merger_root (dict):
            dictionary with parameters to be merged into params root;
            overwrites values of params_root for the same keys and level.
        base (bool):
            True for the first level of the recursion
    """
    if base:
        assert isinstance(params_root, dict)
        assert isinstance(merger_root, dict)
        merger_root = copy.deepcopy(merger_root)

    if merger_root:  # non-empty dict
        for key, val in merger_root.items():
            if isinstance(val, dict) and key in params_root:
                merger_root[key] = dict_recursive_merge(params_root[key],
                                                        merger_root[key],
                                                        base=False)

        merger_root = {**params_root, **merger_root}

    return merger_root


def assign_tree_structure(assign_root, structure, base=True):
    """
    Assign a tree structure to dictionary according to structure.
    (see structure variable in default_struct_values.py)

    Args:
        assign_root (dict):
            dictionary to be assigned a tree structure
        base (bool):
            True for the first level of the recursion
    """
    if base:
        assert isinstance(assign_root, dict)
        assert isinstance(structure, dict)
        assign_root = copy.deepcopy(assign_root)
        global assign_orig
        assign_orig = assign_root  # keep the original dict
        global leaves
        leaves = []  # leaves on dict tree levels deeper than base

    if structure:  # non-empty dict
        for key, val in structure.items():
            if isinstance(val, dict):
                assign_root[key] = {}
                assign_root[key] = assign_tree_structure(assign_root[key],
                                                         structure[key],
                                                         base=False)
                if len(assign_root[key]) < 1:
                    # do not have empty dictionaries in assign_root
                    del assign_root[key]
            else:
                assert val is None, 'leaf values in structure should be None'
                if key in assign_orig:
                    assign_root[key] = assign_orig[key]
                    if not base:
                        leaves.append(key)

    # delete duplicated leaves in base root if they are not in first level of
    # structure dict
    if base:
        for key in leaves:
            if key not in structure and key in assign_root:
                del assign_root[key]

    return assign_root


def flatten_structure(assign_root, base=True):
    """
    Reverses the operation of assign_tree_structure()
    by flattening input dictionary.

    Args:
        assign_root (dict):
            dictionary to be flattened
        base (bool):
            True for the first level of the recursion
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
    """ Check if tensors are in device 'dev' """

    for t in tensors:
        assert t.device == torch.device(dev), f'tensor is not in device {dev}'


def spline_grid_from_range(spline_size, spline_range, round_to=1e-6):
    """
    Compute spline grid spacing from desired one-sided range
    and the number of activation coefficients.

    Args:
        spline_size (odd int):
            number of spline coefficients
        spline_range (float):
            one-side range of spline expansion.
        round_to (float):
            round grid to this value
    """
    if int(spline_size) % 2 == 0:
        raise TypeError('size should be an odd number.')
    if float(spline_range) <= 0:
        raise TypeError('spline_range needs to be a positive float...')

    spline_grid = ((float(spline_range) /
                    (int(spline_size) // 2)) // round_to) * round_to

    return spline_grid


def init_sub_dir(top_dir, sub_dir_name):
    """
    Returns the sub-directory folder 'top_dir/sub_dir_name'.
    Creates it if it does not exist.
    """
    sub_dir = os.path.join(top_dir, sub_dir_name)
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)

    return sub_dir


def json_load(json_filename):
    """
    Load a json file.

    Args:
        json_filename (str):
            Path of the .json file with results.
    Returns:
        results_dict (dict):
            dictionary with results stored in the json file.
    """
    try:
        with open(json_filename) as jsonfile:
            results_dict = json.load(jsonfile)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise

    return results_dict


def json_dump(results_dict, json_filename):
    """
    Save results in a json file.

    Args:
        results_dict (dict):
            dictionary with the results to be stored in the json file.
        json_filename (str):
            Path of the .json file where results are stored.
    """
    try:
        with open(json_filename, 'w') as jsonfile:
            json.dump(results_dict, jsonfile, sort_keys=False, indent=4)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise


def add_date_to_filename(filename):
    """
    Adds current date to a filename.

    Args:
        filename (str)
    Returns:
        new_filename (str):
            filename with added date.
    """
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    new_filename = '_'.join([filename, dt_string])

    return new_filename
