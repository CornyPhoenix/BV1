# -*- coding: utf-8 -*-
"""
Module:
=======
Bildverarbeitung 1

Exercise: 
=========
E02 - Image formation

Authors:
========
Toprak,  Sibel      (6712316)
Brand,   Axel       (6145101)
Möllers, Konstantin (6313136)

"""

from __future__ import division
import colorsys

import numpy as np

from scipy import misc
import matplotlib.pyplot as plt


def mandelbrot(width, height, bg_color, fg_color):
    """
    Creates a Mandelbrot set with a specified width and height.
    The background and the foreground color can be given.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    m = width / 2
    n = height / 2
    for y in range(height):
        y2 = (y - n) / height * -2
        for x in range(width):
            x2 = (x - m) / height * 2 - 0.5

            ok = True
            image[y][x] = bg_color

            zx = 0
            zy = 0

            xx = zx * zx
            yy = zy * zy
            xy = zx * zy

            for c in range(19):
                zx = xx - yy + x2
                zy = xy + xy + y2

                xx = zx * zx
                yy = zy * zy
                xy = zx * zy

                if xx + yy > 4:
                    ok = False
                    break

            if ok:
                image[y][x] = fg_color

    return image


def rgb_to_hls(rgb):
    return colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def brightness(rgb):
    # return (min(rgb) + max(rgb)) / 2
    return rgb_to_hls(rgb)[1] * 255


def print_brightness(image):
    target = image.copy()
    for y in range(len(image)):
        for x in range(len(image[y])):
            rgb = image[y, x]
            target[y, x] = brightness(rgb)

    return target


def calibrate_image(image):
    tl = 0.5  # target brightness
    for y in range(len(image)):
        for x in range(len(image[y])):
            h, l, s = rgb_to_hls(image[y, x])

            r, g, b = colorsys.hls_to_rgb(h, tl, s)
            image[y][x] = [r * 255, g * 255, b * 255]

    return image


class E02:
    def __init__(self):
        self.b1 = None
        self.b2 = None

    """
    1. PHOTOMETRY & TV IMAGING
    """

    def exercise1a(self):
        """
        Photometry
        """
        pass

    def exercise1b(self):
        """
        TV Images
        """
        # 5m in reality correspond to 50px on screen:
        pixels_per_meter = 10

        # The Car has a velocity of 50km/h in reality:
        velocity_reality = 50
        print 'Velocity in reality:   ', velocity_reality, '[km/h]'

        # An hour has 3,600 seconds, so 3,600,000 it has ms:
        ms_per_hour = 3600 * 1000

        # We calculate the conversion factor for px/ms on the screen from km/h:
        kmh_to_pxms = 1000 * pixels_per_meter / ms_per_hour

        # The velocity on screen in px/ms is now the converted velocity in reality given in km/h:
        velocity_screen = velocity_reality * kmh_to_pxms
        print 'Velocity on screen:    ', velocity_screen, '[px/ms]'

        # v = s/t. We are looking for s = v * t.
        # Time t is the time that half of a picture needs to get rendered, because we have interlaced mode
        # and so from line 200 to 201 first all rows 202, 204, 206, ..., 574, 576, 1, 3, 5, ..., 197, 199
        # get rendered until we reach line 201, so these are 576 / 2 = 288 rows. The script says, that one
        # line needs 64ms to render.
        time_per_row = 64  # ms
        number_of_rows = 576 / 2
        time = time_per_row * number_of_rows
        print '∆t from l. 200 to 201: ', time, '[ms]'

        # offset = velocity on screen * time difference
        offset = velocity_screen * time

        print 'Offset in Pixels:      ', offset, '[px]'

    """
    2. COLOR PERCEPTION
    """

    def exercise2a(self):
        """
        Generate a color image B1 with non-calibrated colors (i.e. arbitrary colors with different brightness values).
        Determine the brightness values of background and color areas.
        """
        self.b1 = mandelbrot(600, 400, [255, 125, 125], [0, 255, 0])
        plt.axis('off')
        plt.imshow(self.b1)
        plt.show()
        misc.imsave("B1.png", self.b1)
        misc.imsave("B1_Brightness.png", print_brightness(self.b1))

    def exercise2b(self):
        """
        Using B1 as input, generate a calibrated image B2 with equally bright background and color areas.
        """
        self.b2 = calibrate_image(self.b1)
        plt.axis('off')
        plt.imshow(self.b2)
        plt.show()
        misc.imsave("B2.png", self.b2)
        misc.imsave("B2_Brightness.png", print_brightness(self.b2))


if __name__ == '__main__':
    e02 = E02()

    e02.exercise1a()
    e02.exercise1b()
    e02.exercise2a()
    e02.exercise2b()
