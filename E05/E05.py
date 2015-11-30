# -*- coding: utf-8 -*-
"""
Module:
=======
Bildverarbeitung 1

Exercise:
=========
E05 - Image Compression and Segmentation

Authors:
========
Dobert, Tim         (6427948)
Möllers, Konstantin (6313136)

"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy import misc


def eigenvector_matrix(matrix, dims):
    """
    Calculates a matrix of eigenvectors with
    the `dim`-highest eigenvalues.
    :param matrix: Input matrix to calculate eigenvectors from.
    :param dims: Number of dimensions to compute.
    :return: Matrix with `dim` rows.
    """
    width, width = matrix.shape
    ew, ev = la.eig(matrix)

    # Get arguments of eigenvalues in descending order
    sorted_args = np.argsort(ew)[::-1]

    result = np.zeros((dims, width))
    for i in range(dims):
        arg = sorted_args[i]
        result[i] = ev[:, arg].T

    return result


if __name__ == '__main__':
    # Exercise 2.2a)

    # covariance matrix V, taken from the exercise.
    covariance_matrix = 0.25 * np.matrix([
        [  15,  5,  9,  3],
        [   5, 15,  3,  9],
        [   9,  3, 15,  5],
        [   3,  9,  5, 15],
    ])

    # A is the matrix of three eigenvectors of V wit highest eigenvalues
    A = eigenvector_matrix(covariance_matrix, 3)
    print np.dot(A, A.T)
