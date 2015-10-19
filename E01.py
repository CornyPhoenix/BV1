# -*- coding: utf-8 -*-
"""
Module:
=======
Bildverarbeitung 1

Exercise: 
=========
E01 - Getting started

Authors:
========
Toprak,  Sibel      (6712316)
Brand,   Axel       (6145101)
Möllers, Konstantin (6313136)

"""

import numpy
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt

def mirrorVertically(image):
    """
    Vertically mirrors an image.
    """
    result = []
    height = len(image)
    for y in range(height):
        result.append(image[height - y - 1])
        
    return result
    
def mirrorHorizontally(image):
    """
    Horizontally mirrors an image.
    """
    result = []
    for row in image:
        mirroredRow = []
        
        width = len(row)
        for x in range(width):
            mirroredRow.append(row[width - x - 1])
        
        result.append(mirroredRow)
        
    return result

def mirror(image, horizontal, vertical):
    """
    Mirrors an image. Pass boolean parameters to this function.
    """
    if horizontal:
        image = mirrorHorizontally(image)
        
    if vertical:
        image = mirrorVertically(image)
        
    return image

# Example image from scipy
l = misc.lena()
m = mirror(l, horizontal=False, vertical=False)
plt.imshow(m, cmap=plt.cm.gray)
