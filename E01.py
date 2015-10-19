# -*- coding: utf-8 -*-

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

def mirror(image, horizontal):
    """
    Mirrors an image.
    """
    if horizontal:
        return mirrorHorizontally(image)
        
    return mirrorVertically(image)

# Example image from scipy
l = misc.lena()
m = mirror(l, horizontal=False)
plt.imshow(m, cmap=plt.cm.gray)
