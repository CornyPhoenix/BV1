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
Elfaramawy, Nourhan (6517858)
Brand,   Axel       (6145101)
Möllers, Konstantin (6313136)

"""

from __future__ import division
import colorsys

import numpy as np

from scipy import misc
import matplotlib.pyplot as plt


def mandelbrot(width, height, colors, inf_color, zoom = 2, xshift = 0.5, yshift = 0.0):
    """
    Creates a Mandelbrot set with a specified width and height.
    The background and the foreground color can be given.
    """
    # Create an image array.
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    m = int(width / 2)  # Half image width
    n = int(height / 2) # half image height
    
    mx = 200 # Maximum number of iterations
    clrlen = (len(colors) - 1) / mx # multiplicator for the color mapping
    
    # y is a coordinate on the image plane going from top to bottom.
    for y in range(height):
        # y2 is a coordinate on the Mandelbrot plane going from bottom to top.
        y2 = (n - y) / height * zoom - yshift
        
        # x is a coordinate on the image plane going from left to right.
        for x in range(width):
            # x2 is a coordinate on the Mandelbrot plane going from left to right.
            x2 = (x - m) / height * zoom - xshift

            # Reset the Mandelbrot iterations
            zx = 0
            zy = 0

            # Precalculations of products for first iteration
            xx = zx * zx
            yy = zy * zy
            xy = zx * zy

            # Use infinity color first.
            image[y][x] = inf_color
            
            for c in range(mx):
                zx = xx - yy + x2
                zy = xy + xy + y2

                # Precalculations of products for next iteration
                xx = zx * zx
                yy = zy * zy
                xy = zx * zy

                # Do the coordinates leave the allowed circle?
                # Pick a color for the number of iterations.
                if xx + yy > 4:
                    image[y][x] = colors[int(round(c * clrlen))]
                    break

    return image


def rgb_to_hls(rgb):
    """
    Converts an array with RGB values to the HLS color model.
    """
    return colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def brightness(rgb):
    """
    Returns a greyvalue in 0..255 that can be used for an image pixel.
    """
    # return (min(rgb) + max(rgb)) / 2
    return rgb_to_hls(rgb)[1] * 255


def print_brightness(image):
    """
    Creates a brightness showing greyscale image from an input.
    """
    target = image.copy()
    for y in range(len(image)):
        for x in range(len(image[y])):
            rgb = image[y, x]
            target[y, x] = brightness(rgb)

    return target


def calibrate_image(image, tl = 0.5):
    """
    Calibrates an image so that all colors get the same target brightness
    value `tl` (defaults to 0.5).
    """
    for y in range(len(image)):
        for x in range(len(image[y])):
            # Convert the exiting pixel color to HLS model.
            h, l, s = rgb_to_hls(image[y, x])

            # We convert the HLS model back to RGB, but using our target 
            # brightness value `tl`.
            r, g, b = colorsys.hls_to_rgb(h, tl, s)
            
            # Set the new pixel color.
            image[y][x] = [r * 255, g * 255, b * 255]

    return image


class E02:
    def __init__(self):
        self.b1 = None
        self.b2 = None

    def exercise1b(self):
        """
        TV Images
        """
        # 5m in reality correspond to 50px on screen:
        pixels_per_meter = 10

        # The Car has a velocity of 50km/h in reality:
        velocity_reality = 50
        print 'Velocity in reality:   ', velocity_reality, '[km/h]'

        # An hour has 3,600 seconds, so 3,600,000,000 it has ms:
        micros_per_hour = 3600 * 1000000

        # We calculate the conversion factor for px/ms on the screen from km/h:
        kmh_to_pxmicros = 1000 * pixels_per_meter / micros_per_hour

        # The velocity on screen in px/ms is now the converted velocity in reality given in km/h:
        velocity_screen = velocity_reality * kmh_to_pxmicros
        print 'Velocity on screen:    ', velocity_screen, '[px/μs]'

        # v = s/t. We are looking for s = v * t.
        # Time t is the time that half of a picture needs to get rendered, because we have interlaced mode
        # and so from line 200 to 201 first all rows 202, 204, 206, ..., 574, 576, 1, 3, 5, ..., 197, 199
        # get rendered until we reach line 201, so these are 576 / 2 = 288 rows. The script says, that one
        # line needs 64ms to render.
        time_per_row = 64  # ms
        number_of_rows = 576 / 2
        time = time_per_row * number_of_rows
        print '∆t from l. 200 to 201: ', time, '[μs]'

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
        colors = [
            [253, 253, 150],
            [255, 125, 125],
            [150, 111, 214],
            [119, 158, 203],
            [255, 105,  97],
            [  3, 192,  60]
        ]
        # This takes some time …
        #self.b1 = mandelbrot(1200, 600, colors, [0, 0, 0], 0.02, 0.73, 0.21)
        # Load the pregenerated image ;)
        self.b1 = misc.imread("B1.png")
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

    e02.exercise1b()
    e02.exercise2a()
    e02.exercise2b()
