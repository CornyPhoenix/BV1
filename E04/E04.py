# -*- coding: utf-8 -*-
"""
Module:
=======
Bildverarbeitung 1

Exercise:
=========
E04 - Histograms, Filters, Convolution and Fourier-Transform

Authors:
========
Dobert, Tim         (6427948)
Möllers, Konstantin (6313136)

"""

from __future__ import division

from math import log

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2
from scipy import misc


def histogram(image):
    """
    Builds the histogram for an image.

    :param image: The image to construct the histogram for.
    :return: An array containing all gray values.
    """
    width, height = np.shape(image)
    result = np.zeros(256)
    for y in range(height):
        for x in range(width):
            result[image[y, x]] += 1
    return result


def histogram_to_image(histo):
    height = int(max(histo) / 4) + 10
    bar_width = 5
    width = 255 * bar_width
    # Building the image
    img = []
    for i in range(height):
        img.append([0 for i in range(width)])
    for x in range(width):
        for y in range(int(histo[int(x / bar_width)] / 4)):
            img[height - y - 1][x] = 255
    return img


def normalize_gray_values(image):
    """
    Normalizes gray values of an image.

    :param image: The image to normalize.
    :return: A copy of the image with normalized gray values.
    """
    # Get image dimensions
    width, height = np.shape(image)

    std = np.std(image)  # Standardabweichung
    mean = np.mean(image)  # Mittelwert

    # Calculate boundaries for which we normalize the data
    max_black = mean - 1.644854 * std  # ca. 90% of everything (95% of a half)
    min_white = mean + 1.281552 * std  # ca. 80% of everything (90% of a half)

    # Linear coefficient to transform other pixels
    coefficient = 255 / (255 - max_black - (255 - min_white))

    output = image.copy()
    for row in range(height):
        for cell in range(width):
            value = image[row, cell]

            # Make 5% of darkest pixels black
            if value <= max_black:
                output[row, cell] = 0
            # Make 10% of lightest pixels white
            elif value >= min_white:
                output[row, cell] = 255
            # Linear transform others
            else:
                output[row, cell] = (value - max_black) * coefficient

    return output


def apply_func_2d(image, func):
    """
    Applies a one dimensional function "func" on a two dimensional image.

    :param image: The image to transform.
    :param func: The function to apply.
    :return: The result from applying the function in two dimensions.
    """
    # Get image dimensions
    width, height = np.shape(image)

    # Generate a matrix for complex numbers with same dimensions as
    # the input image and apply the function on each row.
    rows = np.zeros((height, width), 'complex')
    for y in range(height):
        row = image[y:y + 1, :]
        rows[y, :] = func(row)

    # Afterwards, apply the function on each column.
    cols = np.zeros((height, width), 'complex')
    for x in range(width):
        col = rows[:, x:x + 1].T
        cols[:, x] = func(col).T[:, 0]

    return cols


def fourier_transform(spatial):
    """
    Applies the 1D DFT as 2D function.

    :param spatial: The image values coming from the spatial domain.
    :return: The image values in the frequency domain.
    """
    return apply_func_2d(spatial, fft)


def rev_fourier_transform(frequency):
    """
    Applies the 1D inverse DFT as 2D function.

    :param frequency: The image values coming from the frequency domain.
    :return: The image values in the spatial domain.
    """
    return apply_func_2d(frequency, ifft).real


def convolve(image1, image2, MinPad=True, pad=True):
    """ Not so simple convolution """

    # The size of the images:
    r1, c1 = image1.shape
    r2, c2 = image2.shape

    # MinPad results simpler padding,smaller images:
    if MinPad:
        r = r1 + r2
        c = c1 + c2
    else:
        # if the Numerical Recipies says so:
        r = 2 * max(r1, r2)
        c = 2 * max(c1, c2)

    # For nice FFT, we need the power of 2:
    if pad:
        pr2 = int(log(r) / log(2.0) + 1.0)
        pc2 = int(log(c) / log(2.0) + 1.0)
        rOrig = r
        cOrig = c
        r = 2 ** pr2
        c = 2 ** pc2
    # end of if pad

    # numpy fft has the padding built in, which can save us some steps
    # here. The thing is the s(hape) parameter:
    fftimage = fft2(image1, s=(r, c)) * fft2(image2, s=(r, c))

    if pad:
        return (ifft2(fftimage))[:rOrig, :cOrig].real
    else:
        return (ifft2(fftimage)).real


def plot_gray(image):
    """
    Plots a gray value image.

    :param image: The image to plot.
    """
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':

    # Exercise 1d)
    b1 = misc.imread("B1.png", flatten=True)
    b2 = convolve(b1, b1)
    misc.imsave("B2.png", b2)

    # Exercise 2a)
    lena = misc.lena()

    normalized_lena = normalize_gray_values(lena)
    misc.imsave("normalized_lena.png", normalized_lena)

    # Histograms:
    misc.imsave("histogram_lena.png", histogram_to_image(histogram(lena)))
    misc.imsave("histogram_normalized_lena.png", histogram_to_image(histogram(normalized_lena)))

    # Exercise 2b)
    fft_image = fourier_transform(lena)
    rev_image = rev_fourier_transform(fft_image)
    misc.imsave("rev_image.png", rev_image)
