#!/usr/bin/env python3
'''
This script is illustrative. It plots an example of a deepspline
along with its B-spline and boundary basis elements.
Please run script with --help for argument details.
'''

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def Bspline(x, center=0, grid=1, coeff=1, mode='both'):
    """
    Evaluates a B-spline basis element at x.

    Args:
        x (np.array): input locations.
        center (float): center of the basis function.
        grid (float): grid spacing (determines width of B-spline).
        coeff (float): coefficient of the B-spline (height).

    Returns:
        y (np.array): of the same size as x.
    """
    assert mode in ['both', 'left', 'right']
    y = np.zeros(x.shape)

    if not (mode == 'right'):
        left_idx = (x > (center - grid)) * (x <= center)
        y[left_idx] = (x[left_idx] - (center - grid)) / grid

    if not (mode == 'left'):
        right_idx = (x < (center + grid)) * (x >= center)
        y[right_idx] = ((center + grid) - x[right_idx]) / grid

    return y * coeff  # basis * coefficient


def plot_deepspline_basis(args):
    """
    Args:
        args: verified arguments from arparser
    """
    plt.figure()
    ax = plt.gca()

    # (B-spline expansion range, grid spacing,
    #  nb. plot points, extrapolation range)
    range_, grid, nb_points, extrap = 3, 1, 10001, 2
    # the total plot x axis range is then [-5, 5] = [-(range_+extrap),
    # (range+extrap)]

    # for B-spline expansion
    x_middle = np.linspace(-range_, range_, nb_points)
    # for linear extrapolations outside B-spline range
    x_left = np.linspace(-(range_ + extrap), -range_, nb_points)
    x_right = np.linspace(range_, (range_ + extrap), nb_points)

    # grid for plotting B-spline elements in [-3, 3]
    grid_points = np.arange(-range_ - 1, range_ + 2, grid)
    # grid for plotting boundary elements in [-5, -3] and [3, 5]
    left_grid_points = np.arange(-(range_ + extrap), -range_ + 1, grid)
    right_grid_points = np.arange(range_, range_ + extrap + 1, grid)

    # B-spline coefficients
    coeff = np.array([4.5, 3.3, 5.3, 2.3, 3.3, 1.3, 4.5, 3.5, 3.1])

    # left and right linear extrapolations at grid locations in [-5, -3] and
    # [3, 5]
    left_extrap = (coeff[0] - coeff[1]) * \
        np.array(list(range(0, extrap + 1)))[::-1] + coeff[1]
    right_extrap = (coeff[-1] - coeff[-2]) * \
        np.array(list(range(0, extrap + 1))) + coeff[-2]

    # values of boundary basis at grid locations in [-5, -3] and [3, 5]
    right_straight = np.ones(extrap + 1) * coeff[-2]
    left_straight = np.ones(extrap + 1) * coeff[1]
    left_relu = (coeff[0] - coeff[1]) * \
        np.array(list(range(0, extrap + 1)))[::-1]
    right_relu = (coeff[-1] - coeff[-2]) * np.array(list(range(0, extrap + 1)))

    # B-spline expansion function
    f = interp1d(grid_points, coeff)

    # extrapolation functions
    f_left = interp1d(left_grid_points, left_extrap)
    f_right = interp1d(right_grid_points, right_extrap)

    # boundary functions
    f_left_straight = interp1d(left_grid_points, left_straight)
    f_right_straight = interp1d(right_grid_points, right_straight)
    f_left_relu = interp1d(left_grid_points, left_relu)
    f_right_relu = interp1d(right_grid_points, right_relu)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
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
                       fontdict={
                           'horizontalalignment': 'center',
                           'fontsize': 10},
                       minor=False)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    # draw B-spline (triangular-shaped) basis elements
    for i, center in enumerate(grid_points):
        mode = 'both'
        if i == 0 or i == (grid_points.shape[0] - 1):
            # skip (boundaries)
            continue
        elif i == 1:
            # first B-spline basis: only right part is drawn
            mode = 'right'
        elif i == (grid_points.shape[0] - 2):
            # last B-spline basis: only left part is drawn
            mode = 'left'

        # evaluate B-spline basis element on a grid
        bspline_x = np.linspace(-(range_ + 2) + i * grid,
                                -(range_ + 2) + (i + 2) * grid, nb_points)
        bspline_y = Bspline(bspline_x, center, grid, coeff[i], mode=mode)

        if mode == 'left':
            center_idx = bspline_x.shape[0] // 2
            # draws right part of first B-spline basis elemnt
            plt.plot(bspline_x[:center_idx:],
                     bspline_y[:center_idx:],
                     color='lightsteelblue',
                     ls='--')
        elif mode == 'right':
            center_idx = bspline_x.shape[0] // 2
            # draws left part of first B-spline basis elemnt
            plt.plot(bspline_x[center_idx::],
                     bspline_y[center_idx::],
                     color='lightsteelblue',
                     ls='--')
        else:
            # draws full B-spline basis elemnt
            plt.plot(bspline_x, bspline_y, color='crimson', ls='--')

    # plot B-spline expansion
    plt.plot(x_middle, f(x_middle), color='black')
    # plot linear extrapolations
    plt.plot(x_left, f_left(x_left), color='black')
    plt.plot(x_right, f_right(x_right), color='black')

    # plot boundary elements
    plt.plot(x_left, f_left_straight(x_left), color='lightsteelblue', ls='--')
    plt.plot(x_right,
             f_right_straight(x_right),
             color='lightsteelblue',
             ls='--')
    plt.plot(x_left, f_left_relu(x_left), color='lightsteelblue', ls='--')
    plt.plot(x_right, f_right_relu(x_right), color='lightsteelblue', ls='--')

    plt.xlim(-(range_ + extrap - 0.2), (range_ + extrap - 0.2))
    plt.ylim(-0.8, 5.5)
    plt.gca().set_position([0, 0, 1, 1])

    if args.save_dir is not None:
        plt.savefig(os.path.join(args.save_dir, 'deepspline_basis.png'))

    plt.show()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Plot finite spline representation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--save_dir',
        metavar='[STR]',
        type=str,
        help='directory for saving plots. If not given, plots are not saved.')

    args = parser.parse_args()

    if args.save_dir is not None and not os.path.isdir(args.save_dir):
        raise OSError(f'Save directory {args.save_dir} does not exist.')

    plot_deepspline_basis(args)
