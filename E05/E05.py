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


def hue_to_rgb(hue, saturation):
    """
    Converts a HSV value to an RGB array.
    """
    while hue < 0:
        hue += 2 * math.pi
    while hue > 2 * math.pi:
        hue -= 2 * math.pi

    k = 3 * hue / math.pi
    i = math.floor(k)
    f = (k - i)
    p = 255 * (1 - saturation)
    q = 255 * (1 - saturation * f)
    t = 255 * (1 - saturation * (1 - f))

    if i == 0 or i == 6:
        return [255, t, p]

    if i == 1:
        return [q, 255, p]

    if i == 2:
        return [p, 255, t]

    if i == 3:
        return [p, q, 255]

    if i == 4:
        return [t, p, 255]

    return [255, p, q]


class EdgeDetectionResult:
    def __init__(self, magnitudes, angles):
        self.angles = angles
        self.magnitudes = magnitudes

    def magnitudes_image(self):
        """
        Creates an image of the magnitudes.
        """
        return Image(self.magnitudes * 255)

    def angles_image(self):
        """
        Creates an image of the angles.
        """
        height, width = self.angles.shape
        img = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                img[y, x, :] = hue_to_rgb(self.angles[y, x], abs(self.magnitudes[y, x]))

        return Image(img)


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
        misc.imsave(filename, self.image)

    def roberts_cross(self):
        """
        Calculates the Robert's Cross operator on this image.
        :return: (magnitudes, angles)
        """

        # Normalize image to 0..1
        g = self.image.copy() / 255

        # Get image dimensions and create resulting images
        shape = g.shape
        magnitudes = np.zeros(shape)
        angles = np.zeros(shape)
        height, width = shape

        # Some helper vars for optimization
        sqrt2 = math.sqrt(2)

        for y in range(height):
            y1 = max(y - 1, 0)
            y2 = y
            for x in range(width):
                x1 = max(x - 1, 0)
                x2 = x

                delta_1 = g[y1, x2] - g[y2, x1]
                delta_2 = g[y1, x1] - g[y2, x2]

                # Save normalized (0..1) magnitude
                magnitudes[y, x] = math.sqrt(delta_1 ** 2 + delta_2 ** 2) / sqrt2

                # Calculate angle if possible
                if (g[y1, x2] - g[y2, x1]) != 0:
                    angles[y, x] = math.atan2(delta_2, delta_1)

        return EdgeDetectionResult(magnitudes, angles)

    def sobel(self):
        """
        Calculates the Sobel operator on this image.
        :return: (magnitudes, angles)
        """

        # Normalize image to 0..1
        g = self.image.copy() / 255

        # Get image dimensions and create resulting images
        shape = g.shape
        magnitudes = np.zeros(shape)
        angles = np.zeros(shape)
        height, width = shape

        # Some helper vars for optimization
        sqrt32 = math.sqrt(2)

        for y in range(height):
            y1 = max(y - 1, 0)
            y2 = y
            y3 = min(y + 1, height - 1)
            for x in range(width):
                x1 = max(x - 1, 0)
                x2 = x
                x3 = min(x + 1, width - 1)

                # delta_x in -4..4
                delta_x = g[y1, x3] + 2 * g[y2, x3] + g[y3, x3] \
                        - g[y1, x1] - 2 * g[y2, x1] - g[y3, x1]

                # delta_y in -4..4
                delta_y = g[y3, x1] + 2 * g[y3, x2] + g[y3, x3] \
                        - g[y1, x1] - 2 * g[y1, x2] - g[y1, x3]

                # magnitude in -sqrt32..sqrt32
                magnitude = math.sqrt(delta_x ** 2 + delta_y ** 2)
                magnitudes[y, x] = magnitude / sqrt32
                if delta_x != 0:
                    angles[y, x] = math.atan2(delta_y, delta_x)

        return EdgeDetectionResult(magnitudes, angles)

    def kirsch(self):
        """
        Calculates the Kirsch operator on this image.
        :return: (magnitudes, angles)
        """

        # Normalize image to 0..1
        g = self.image.copy() / 255

        # Get image dimensions and create resulting images
        shape = g.shape
        magnitudes = np.zeros(shape)
        angles = np.zeros(shape)
        height, width = shape

        # Some helper vars for optimization
        ks = range(8)  # 0..7
        mod = lambda k: k % 8
        limit = lambda k, mn, mx: mn if (k < mn) else mx if (k > mx) else k

        ys = [0, 1, 1, 1, 0, -1, -1, -1]  # Shift in Y direction
        xs = [1, 1, 0, -1, -1, -1, 0, 1]  # Shift in X direction
        sqrt32 = math.sqrt(2)

        for y in range(height):
            for x in range(width):
                gk = [g[limit(y + ys[k], 0, height - 1), limit(x + xs[k], 0, width - 1)] for k in ks]
                magns = [3 * (gk[k] + gk[mod(k + 1)] + gk[mod(k + 2)] + gk[mod(k + 3)] + gk[mod(k + 4)]) - 5 * (gk[mod(k + 5)] + gk[mod(k + 6)] + gk[mod(k + 7)]) for k in ks]
                k_max = np.argsort(magns)[7]

                magnitudes[y, x] = magns[k_max] / 15
                angles[y, x] = math.pi * ((2 + k_max) % 8) / 4

        return EdgeDetectionResult(magnitudes, angles)

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

    print('Appyling Robert\'s Cross operator ...'),
    roberts_cross = lena.roberts_cross()
    roberts_cross.magnitudes_image().save("roberts_cross_magnitudes.png")
    roberts_cross.angles_image().save("roberts_cross_angles.png")
    print('done.')

    print('Appyling Sobel operator ...'),
    sobel = lena.sobel()
    sobel.magnitudes_image().save("sobel_magnitudes.png")
    sobel.angles_image().save("sobel_angles.png")
    print('done.')

    print('Appyling Kirsch operator ...'),
    kirsch = lena.kirsch()
    kirsch.magnitudes_image().save("kirsch_magnitudes.png")
    kirsch.angles_image().save("kirsch_angles.png")
    print('done.')

