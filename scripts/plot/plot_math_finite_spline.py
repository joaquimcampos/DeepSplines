#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import interp1d
import argparse


def triangle(x, center=0, grid=1, coeff=1, mode='both'):
    assert mode in ['both', 'left', 'right']
    y = np.zeros(x.shape)

    if not (mode == 'right'):
        left_idx  = (x > (center-grid)) * (x <= center)
        y[left_idx]  = (x[left_idx]   - (center-grid)) / grid

    if not (mode == 'left'):
        right_idx = (x < (center+grid)) * (x >= center)
        y[right_idx] = ((center+grid) - x[right_idx])  / grid

    return y*coeff


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Plot finite spline representation.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--savefig', action='store_true', help='')
    parser.add_argument('--output', metavar='output folder', type=str, help='')
    args = parser.parse_args()

    fig = plt.figure()
    ax = plt.gca()

    range_, grid, nb_points, extrap = 4, 1, 10001, 1

    x_middle = np.linspace(-range_+1, range_-1, nb_points)
    x_left = np.linspace(-(range_ + extrap), -range_+1, nb_points)
    x_right = np.linspace(range_-1, (range_ + extrap), nb_points)

    grid_points = np.arange(-range_, range_ + 1, grid)
    left_grid_points = np.arange(-(range_ + extrap), -range_+2, grid)
    right_grid_points = np.arange(range_-1, range_ + extrap + 1, grid)

    coeff = np.array([4.5, 3.3, 5.3, 2.3, 3.3, 1.3, 4.5, 3.5, 3.1])
    left_extrap = (coeff[0] - coeff[1]) * np.array(list(range(0, extrap+2)))[::-1] + coeff[1]
    right_extrap = (coeff[-1] - coeff[-2]) * np.array(list(range(0, extrap+2))) + coeff[-2]

    right_straight = np.ones(extrap + 2) * coeff[-2]
    left_straight = np.ones(extrap + 2) * coeff[1]
    left_relu = (coeff[0] - coeff[1]) * np.array(list(range(0, extrap+2)))[::-1]
    right_relu = (coeff[-1] - coeff[-2]) * np.array(list(range(0, extrap+2)))

    f_left = interp1d(left_grid_points, left_extrap)
    f = interp1d(grid_points, coeff)
    f_right = interp1d(right_grid_points, right_extrap)

    f_left_straight = interp1d(left_grid_points, left_straight)
    f_right_straight = interp1d(right_grid_points, right_straight)
    f_left_relu = interp1d(left_grid_points, left_relu)
    f_right_relu = interp1d(right_grid_points, right_relu)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([1, 2, 4, 5])
    ax.set_xticks([-4, -3, -2, -1, 1, 2, 3, 4])
    ax.set_xticklabels(np.concatenate((np.arange(-4, 0), np.arange(1, 5))),
                fontdict={'horizontalalignment': 'center', 'fontsize': 10}, minor=False)

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    for i, center in enumerate(grid_points):
        mode = 'both'
        if i == 0 or i == (grid_points.shape[0] - 1):
            continue
        elif i == 1:
            mode = 'right'
        elif i == (grid_points.shape[0] - 2):
            mode = 'left'

        x = np.linspace(-(range_+1) + i * grid, -(range_+1) + (i + 2) * grid, nb_points)
        y = triangle(x, center, grid, coeff[i], mode=mode)

        if mode =='left':
            center_idx = x.shape[0]//2
            plt.plot(x[:center_idx:], y[:center_idx:], color='lightsteelblue', ls='--')
        elif mode == 'right':
            center_idx = x.shape[0]//2
            plt.plot(x[center_idx::], y[center_idx::], color='lightsteelblue', ls='--')
        else:
            plt.plot(x, y, color='crimson', ls='--')


    plt.plot(x_middle, f(x_middle), color='black')
    plt.plot(x_left, f_left(x_left), color='black')
    plt.plot(x_right, f_right(x_right), color='black')

    plt.plot(x_left, f_left_straight(x_left), color='lightsteelblue', ls='--')
    plt.plot(x_right, f_right_straight(x_right), color='lightsteelblue', ls='--')

    plt.plot(x_left, f_left_relu(x_left), color='lightsteelblue', ls='--')
    plt.plot(x_right, f_right_relu(x_right), color='lightsteelblue', ls='--')

    plt.xlim(-(range_+extrap-0.2), (range_+extrap-0.2))
    plt.ylim(-0.8, 5.5)

    if args.savefig:
        plt.savefig(os.path.join(args.output, 'finite_spline_representation') + '.pdf')

    plt.show()
