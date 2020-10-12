#!/usr/bin/env python3

import numpy as np
from scipy.linalg import toeplitz


def get_P(T=0.1, size=21):
    """ initialize matrix that transforms deepRelu coefficients (b0, b1, (a))
    into b-spline coefficients (c).
    """
    L = size//2
    A_first_row = np.zeros(size-2) # size: num_slopes
    A_first_row[0] = 1.
    A_first_col = np.arange(1, size-1)
    # construct A toeplitz matrix
    A = toeplitz(A_first_col, A_first_row)
    P_first_col = np.ones((size, 1))
    P_second_col = T*np.arange(-L, L+1)[:, np.newaxis]
    P_zeros = np.zeros((2, size-2))

    P_right = np.concatenate((P_zeros, T*A), axis=0)
    P = np.concatenate((P_first_col, P_second_col, P_right), axis=1)

    return P


def get_P_inv(T=0.1, size=21):
    """ initialize matrix that transforms b-spline coefficients (c)
    into deepRelu coefficients (b0, b1, (a)).
    """
    L = size//2
    D_first_row = np.concatenate((np.array([1., -2., 1.]), np.zeros(size-3)))
    D_first_col = np.zeros(size-2)
    D_first_col[0] = 1
    # construct A toeplitz matrix
    D = toeplitz(D_first_col, D_first_row)

    P_inv_first_row = np.zeros((1, size))
    P_inv_first_row[:, 0] = (1-L)*T
    P_inv_first_row[:, 1] = L*T

    P_inv_second_row = np.zeros((1, size))
    P_inv_second_row[:, 0] = -1
    P_inv_second_row[:, 1] = 1

    P_inv = (1/T) * np.concatenate((P_inv_first_row, P_inv_second_row, D), axis=0)

    return P_inv


if __name__ == "__main__":

    T = 0.1
    size = 21 # 1001

    P = get_P(T, size)
    P_inv = get_P_inv(T, size)

    P_P_inv = P @ P_inv
    P_P_inv[np.absolute(P_P_inv) < 1e-7] = 0.

    P_inv_P = P_inv @ P
    P_inv_P[np.absolute(P_inv_P) < 1e-7] = 0.

    I = np.eye(size)
    assert np.allclose(P_inv_P, I), 'P @ P^-1 is not identity.'
    assert np.allclose(P_P_inv, I), 'P^-1 @ P is not identity.'
