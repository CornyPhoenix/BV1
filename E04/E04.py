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

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2
from scipy import misc


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
    def from_histogram(histogram):
        """
        Creates an image from a histogram.

        :param histogram: a histogram to render.
        :return: a new image instance.
        """
        height = 5120  # Fixed to avoid outliers.
        width = len(histogram)

        # Create an image pane
        img = np.zeros((height, width))

        # Building the image
        for x in range(width):
            value = min(height - 1, histogram[x])
            for y in range(value):
                img[height - 1 - y, x] = 255

        y_scale = .25
        x_scale = 5
        output_shape = (int(y_scale * height), int(x_scale * width))
        return Image(misc.imresize(img, output_shape, 'nearest'))

    @staticmethod
    def from_frequency_domain(frequency):
        """
        Applies the 1D inverse DFT as 2D function.

        :param frequency: The image values coming from the frequency domain.
        :return: The image values in the spatial domain.
        """
        return Image(apply_func_2d(frequency, ifft).real)

    @staticmethod
    def from_lena():
        """
        :return: instance of the image showing Lena Söderberg from the 1972 US-american Playboy.
        """
        return Image(misc.lena())

    def plot_gray(self):
        """
        Plots a gray value image.
        """
        plt.axis('off')
        plt.imshow(self.image, cmap=plt.cm.gray)
        plt.show()

    def save(self, filename):
        """
        Saves the image to file.

        :param filename: Filename for the saved image.
        """
        misc.imsave(filename, self.image)

    def histogram(self):
        """
        Builds the histogram for this image.

        :return: An array containing all gray values.
        """
        width, height = np.shape(self.image)
        result = np.zeros(256, np.int32)
        for y in range(height):
            for x in range(width):
                result[self.image[y, x]] += 1

        return Image.from_histogram(result)

    def convolve(self):
        """
        Convolution method with respect to padding.

        :param other: The other image to convolve with this.
        """
        # The size of the images.
        height, width = self.image.shape

        # Padding needed to move the expected peak to the center.
        # As we convolve with the same image, we just multiply by 2.
        height *= 2
        width *= 2

        # Convolution in the image domain is multiplication in the frequency domain,
        # so we will apply a fft2 first and multiply then.
        frequency_values = fft2(self.image, s=(height, width))
        fft_image = frequency_values * frequency_values

        # Afterwards we transform back to the spatial domain.
        spatial_values = ifft2(fft_image).real

        # Remove the padding for the resulting image so it obtains correct size.
        # (We also box the image data in our Image class)
        return Image(spatial_values[height / 4:3 * height / 4, width / 4:3 * width / 4])

    def normalize_gray_values(self):
        """
        Normalizes the gray values of this image.

        :return: A copy of the image with normalized gray values.
        """
        # Get image dimensions
        width, height = np.shape(self.image)

        std = np.std(self.image)  # Standardabweichung
        mean = np.mean(self.image)  # Mittelwert

        # Calculate boundaries for which we normalize the data
        max_black = mean - 1.644854 * std  # ca. 90% of everything (95% of a half)
        min_white = mean + 1.281552 * std  # ca. 80% of everything (90% of a half)

        # Linear coefficient to transform other pixels
        coefficient = 255 / (255 - max_black - (255 - min_white))

        output = self.image.copy()
        for row in range(height):
            for cell in range(width):
                value = self.image[row, cell]

                # Make 5% of darkest pixels black
                if value <= max_black:
                    output[row, cell] = 0
                # Make 10% of lightest pixels white
                elif value >= min_white:
                    output[row, cell] = 255
                # Linear transform others
                else:
                    output[row, cell] = (value - max_black) * coefficient

        return Image(output)

    def fourier_transform(self):
        """
        Applies the 1D DFT as 2D function.

        :return: The image values in the frequency domain.
        """
        return apply_func_2d(self.image, fft)


if __name__ == '__main__':
    # Exercise 1d)
    b1 = Image.from_file("B1.png")
    b2 = b1.convolve()
    b2.save("B2.png")

    # Exercise 2a)
    lena = Image.from_lena()

    normalized_lena = lena.normalize_gray_values()
    normalized_lena.save("normalized_lena.png")

    # Histograms:
    lena.histogram().save("histogram_lena.png")
    normalized_lena.histogram().save("histogram_normalized_lena.png")

    # Exercise 2b)
    fft_image = lena.fourier_transform()
    rev_image = Image.from_frequency_domain(fft_image)
    rev_image.save("rev_image.png")
