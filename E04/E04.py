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

import numpy as np
import math

from scipy import misc
import matplotlib.pyplot as plt

def build_histogramm(img):
    histo = [0 for x in range(256)]
    for i in range(len(img)):
        for j in range(len(img[0])):
            histo[img[i][j]] += 1
    return histo

def histogram_to_image(histo):
    height = int(max(histo)/4) + 10
    bar_width = 5
    width = 255 * bar_width
    #Building the image
    img = []
    for i in range(height):
        img.append([0 for i in range(width)])
    for x in range(width):
        for y in range(int(histo[int(x/bar_width)]/4)):
            img[height - y -1][x] = 255
    return img

def normalize_gray_values(image):
    """
    Normalizes gray values of an image.

    :param image: The image to normalize.
    :return: A copy of the image with normalized gray values.
    """
    # Get image dimensions
    width, height = np.shape(image)
    
    std = np.std(image)    # Standardabweichung
    mean = np.mean(image)  # Mittelwert

    # Calculate boundaries for which we normalize the data
    max_black = mean - 1.644854 * std # ca. 90% of everything (95% of a half)
    min_white = mean + 1.281552 * std # ca. 80% of everything (90% of a half)

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
    return apply_func_2d(spatial, np.fft.fft)


def rev_fourier_transform(frequency):
    """
    Applies the 1D inverse DFT as 2D function.

    :param frequency: The image values coming from the frequency domain.
    :return: The image values in the spatial domain.
    """
    return apply_func_2d(frequency, np.fft.ifft)


def plot_gray(image):
    """
    Plots a gray value image.

    :param image: The image to plot.
    """
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    lena = misc.lena()
    # plot_gray(lena)

    # normalized_lena = normalize_gray_values(lena)
    # plot_gray(normalized_lena)

    #histograms:
    # plot_gray(histogram_to_image(build_histogramm(lena)))
    # #plot_gray(histogram_to_image(build_histogramm(normalized_lena)))   #weird results


    fft2 = fourier_transform(lena)
    # rev_image = np.fft.ifft2(fft2).real
    rev_image = rev_fourier_transform(fft2)
    plot_gray(rev_image.real)
