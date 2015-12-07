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
from scipy.signal import convolve2d
from scipy.ndimage import filters


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
    def __init__(self, prefix):
        self.prefix = prefix + "_"
        self.images = {}

    def add_grayvalue_image(self, name, grayvalues):
        """
        Creates an image of the magnitudes.
        """
        self.images[name] = Image(grayvalues)

    def add_radians_image(self, name, radians):
        """
        Creates an image of the directions.
        """
        height, width = radians.shape
        img = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                img[y, x, :] = hue_to_rgb(radians[y, x], 1)

        self.images[name] = Image(img)

    def save_images(self):
        for name in self.images.iterkeys():
            self.images[name].save(self.prefix + name + ".png")


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
        :return: (magnitudes, directions)
        """

        roberts_cross_operator = np.array([[+0 - 1j, 1 + 0j],
                                           [-1 + 0j, 0 + 1j]])  # G1 + j*G2
        grad = convolve2d(self.image, roberts_cross_operator, boundary='symm', mode='same')

        result = EdgeDetectionResult("roberts_cross")
        result.add_grayvalue_image("magnitudes", np.absolute(grad))
        result.add_radians_image("directions", np.angle(grad))
        result.add_grayvalue_image("d1", np.real(grad))
        result.add_grayvalue_image("d2", np.imag(grad))

        return result

    def sobel(self):
        """
        Calculates the Sobel operator on this image.
        :return: edge detection result with 4 images
        """

        sobel_operator = np.array([[-1-1j, +0-2j, +1-1j],
                                   [-2+0j, +0+0j, +2+0j],
                                   [-1+1j, +0+2j, +1+1j]])  # Gx + j*Gy
        grad = convolve2d(self.image, sobel_operator, boundary='symm', mode='same')

        result = EdgeDetectionResult("sobel")
        result.add_grayvalue_image("magnitudes", np.absolute(grad))
        result.add_radians_image("directions", np.angle(grad))
        result.add_grayvalue_image("dx", np.real(grad))
        result.add_grayvalue_image("dy", np.imag(grad))

        return result

    def scharr(self):
        """
        Calculates the Scharr operator on this image.
        :return: edge detection result with 4 images
        """

        scharr_operator = np.array([[ -3-3j, 0-10j,  +3 -3j],
                                    [-10+0j, 0+ 0j, +10 +0j],
                                    [ -3+3j, 0+10j,  +3 +3j]])  # Gx + j*Gy
        grad = convolve2d(self.image, scharr_operator, boundary='symm', mode='same')

        result = EdgeDetectionResult("scharr")
        result.add_grayvalue_image("magnitudes", np.absolute(grad))
        result.add_radians_image("directions", np.angle(grad))
        result.add_grayvalue_image("dx", np.real(grad))
        result.add_grayvalue_image("dy", np.imag(grad))

        return result

    def kirsch(self):
        """
        Calculates the Kirsch operator on this image.
        :return: (magnitudes, directions)
        """

        # Normalize image to 0..1
        g = self.image.copy() / 255

        # Get image dimensions and create resulting images
        shape = g.shape
        magnitudes = np.zeros(shape)
        directions = np.zeros(shape)
        height, width = shape

        # Some helper vars for optimization
        ks = range(8)  # 0..7
        mod = lambda k: k % 8
        limit = lambda k, mn, mx: mn if (k < mn) else mx if (k > mx) else k

        ys = [0, 1, 1, 1, 0, -1, -1, -1]  # Shift in Y direction
        xs = [1, 1, 0, -1, -1, -1, 0, 1]  # Shift in X direction

        for y in range(height):
            for x in range(width):
                gk = [g[limit(y + ys[k], 0, height - 1), limit(x + xs[k], 0, width - 1)] for k in ks]
                magns = [3 * (gk[k] + gk[mod(k + 1)] + gk[mod(k + 2)] + gk[mod(k + 3)] + gk[mod(k + 4)]) - 5 * (gk[mod(k + 5)] + gk[mod(k + 6)] + gk[mod(k + 7)]) for k in ks]
                k_max = np.argsort(magns)[7]

                # Calculate direction
                directions[y, x] = math.pi * ((4 + 2 + k_max) % 8) / 4

                # Get highest magnitude
                magnitudes[y, x] = magns[k_max]

        result = EdgeDetectionResult("kirsch")
        result.add_grayvalue_image("magnitudes", magnitudes)
        result.add_radians_image("directions", directions)

        return result

    def laplacian(self):
        """
        Calculates the Laplacian operator on this image.
        :return: (magnitudes, directions)
        """

        laplacian_operator = np.array([[0,  1, 0],
                                       [1, -4, 1],
                                       [0,  1, 0]])
        grad = convolve2d(self.image, laplacian_operator, boundary='symm', mode='same')

        result = EdgeDetectionResult("laplace")
        result.add_grayvalue_image("d2", grad)

        return result


if __name__ == '__main__':
    # Exercise 2.1a)
    print('EXERCISE 2.1')
    print('============')

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
    print("A3 = ", A)
    print("Error = ", error)

    # Exercise 2.2a)
    print('EXERCISE 2.2')
    print('============')
    lena = Image.from_lena()
    lena.save("lena.png")

    print('Applying Robert\'s Cross operator ...'),
    roberts_cross = lena.roberts_cross()
    roberts_cross.save_images()
    print('done.')

    print('Applying Sobel operator ...'),
    sobel = lena.sobel()
    sobel.save_images()
    print('done.')

    print('Applying Kirsch operator ...'),
    kirsch = lena.kirsch()
    kirsch.save_images()
    print('done.')

    print('Applying Scharr operator ...'),
    kirsch = lena.scharr()
    kirsch.save_images()
    print('done.')

    print('Applying Laplacian operator ...'),
    laplacian = lena.laplacian()
    laplacian.save_images()
    print('done.')

