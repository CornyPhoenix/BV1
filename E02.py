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
Toprak,  Sibel      (67.....)
Brand,   Axel       (6......)
Möllers, Konstantin (6313136)

"""

import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt


"""
1. PHOTOMETRY & TV IMAGING
"""


"""
2. COLOR PERCEPTION
"""
data = np.full((200, 200, 3), 255, dtype=numpy.uint8)

for y in range(len(data)):
    for x in range(len(data[y])):
        if y > 50 and y < 150:
            data[y][x] = [255, 125, 125]
        elif x > 50 and x < 150:
            data[y][x] = [255, 125, 0]
        else:
            data[y][x] = [0, 0, 0]
            
def print_brightness(image):
    target = image.copy()
    for y in range(len(image)):
        for x in range(len(image[y])):
            target[y][x] = (min(image[y][x]) + max(image[y][x])) / 2
            
    return target
            
def calibrate_image(image):
    target_v = 127
    for y in range(len(image)):
        for x in range(len(image[y])):
            image[y][x] = [0,0,0]
    

plt.imshow(data)
plt.imshow(print_brightness(data))
