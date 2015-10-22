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
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import colorsys

"""
1. PHOTOMETRY & TV IMAGING
"""


"""
2. COLOR PERCEPTION
"""

#for y in range(len(data)):
#    for x in range(len(data[y])):
#        if y > 50 and y < 150:
#            data[y][x] = [255, 125, 125]
#        elif x > 50 and x < 150:
#            data[y][x] = [255, 125, 0]
#        else:
#            data[y][x] = [0, 0, 0]
            
def mandelbrot(width, height, bg_color, fg_color):
    """
    Creates a Mandelbrot set with a specified width and height.
    The background and the foreground color can be given.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    m = width/2
    n = height/2
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
    #return (min(rgb) + max(rgb)) / 2
    return rgb_to_hls(rgb)[1] * 255
            
def print_brightness(image):
    target = image.copy()
    for y in range(len(image)):
        for x in range(len(image[y])):
            rgb = image[y][x]
            target[y][x] = brightness(rgb)
            
    return target
            
def calibrate_image(image):
    tl = 0.5 # target brightness
    for y in range(len(image)):
        for x in range(len(image[y])):
            h, l, s = rgb_to_hls(image[y][x])
            
            r, g, b = colorsys.hls_to_rgb(h, tl, s)
            image[y][x] = [r * 255, g * 255, b * 255]
                
    return image
            
    
data = mandelbrot(600, 400, [255, 125, 125], [0, 255, 0])
# plt.imshow(data)
misc.imsave("B1.png", data)
misc.imsave("B1_Brightness.png", print_brightness(data))

calibrate_image(data)
misc.imsave("B2.png", data)
misc.imsave("B2_Brightness.png", print_brightness(data))
# plt.imshow(print_brightness(calibrate_image(data)))
