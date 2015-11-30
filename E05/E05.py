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

import numpy as np
import math
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


def mse(matrix_a, matrix_b):
    """
    Calculates the minimum squared error (MSE) of two matrices.
    """
    return ((matrix_a - matrix_b) ** 2).mean(axis=None)


class Image:
    def __init__(self, array):
        self.image = array

    @staticmethod
    def from_file(filename):
        """
        Creates an image from a file.

        :param filename: Filename to obtain the image file.
        :return: image instance.
        """
        return Image(misc.imread(filename, flatten=True))

    @staticmethod
    def from_lena():
        """
        :return: instance of the image showing Lena Söderberg from the 1972 US-american Playboy.
        """
        return Image(misc.lena())

    def save(self, filename):
        """
        Saves the image to file.

        :param filename: Filename for the saved image.
        """
        min = self.image.min()
        delta = self.image.max() - min
        misc.imsave(filename, (self.image - min) / delta * 255)

    def roberts_cross(self):
        """
        Calculates the Robert's Cross operator on this image.
        :return: (magnitudes, angles)
        """

        g = self.image.copy()
        shape = g.shape
        magnitudes = np.zeros(shape)
        angles = np.zeros(shape)
        height, width = shape

        for y in range(height):
            y1 = max(y - 1, 0)
            y2 = y
            for x in range(width):
                x1 = max(x - 1, 0)
                x2 = x

                d1 = g[y1, x2] - g[y2, x1]
                d2 = g[y1, x1] - g[y2, x2]
                magnitudes[y, x] = math.sqrt(d1 ** 2 + d2 ** 2) * np.sign(d1) * np.sign(d2)
                if (g[y1, x2] - g[y2, x1]) != 0:
                    angles[y, x] = math.atan((g[y2, x2] - g[y1, x1]) / (g[y1, x2] - g[y2, x1]))

        return magnitudes, angles

    def sobel(self):
        """
        Calculates the Sobel operator on this image.
        :return: (magnitudes, angles)
        """

        g = self.image.copy()
        shape = g.shape
        magnitudes = np.zeros(shape)
        angles = np.zeros(shape)
        height, width = shape

        for y in range(height):
            y1 = max(y - 1, 0)
            y2 = y
            y3 = min(y + 1, height - 1)
            for x in range(width):
                x1 = max(x - 1, 0)
                x2 = x
                x3 = min(x + 1, width - 1)

                delta_x = g[y1, x3] + 2 * g[y2, x3] + g[y3, x3] \
                        - g[y1, x1] - 2 * g[y2, x1] - g[y3, x1]

                delta_y = g[y3, x1] + 2 * g[y3, x2] + g[y3, x3] \
                        - g[y1, x1] - 2 * g[y1, x2] - g[y1, x3]

                magnitudes[y, x] = math.sqrt(delta_x ** 2 + delta_y ** 2) * np.sign(delta_x) * np.sign(delta_y)
                if delta_x != 0:
                    angles[y, x] = math.atan(delta_y / delta_x)

        return magnitudes, angles

if __name__ == '__main__':
    # Exercise 2.1a)

    # covariance matrix V, taken from the exercise.
    covariance_matrix = 0.25 * np.matrix([
        [  15,  5,  9,  3],
        [   5, 15,  3,  9],
        [   9,  3, 15,  5],
        [   3,  9,  5, 15],
    ])

    # A is the matrix of three eigenvectors of V wit highest eigenvalues
    A = eigenvector_matrix(covariance_matrix, 3)

    # Exercise 2.1b)
    x = np.array([
        [1, 2],
        [3, 4],
    ])
    error = mse(x, np.dot(np.dot(A.T, A), x.reshape(4)).reshape((2, 2)))

    # Exercise 2.2a)
    lena = Image.from_lena()

    roberts_cross_magnitudes, roberts_cross_angles = lena.roberts_cross()
    Image(roberts_cross_magnitudes).save("roberts_cross.png")

    sobel_magnitudes, sobel_angles = lena.sobel()
    Image(sobel_magnitudes).save("sobel.png")

