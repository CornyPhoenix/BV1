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

from scipy import misc
import matplotlib.pyplot as plt

# Example image from scipy
l = misc.lena()
plt.axis('off')
plt.imshow(l, cmap=plt.cm.gray)
plt.show()

"""
2 IMAGE TRANSFORMATION
"""

# For 2a)
def mirror_horizontally(image):
    return image[..., ::-1]
    
def mirror(image):
    """
    Mirrors an image horizontally.
    """
    return mirror_horizontally(image)
    
m = mirror(l)
plt.axis('off')
plt.imshow(m, cmap=plt.cm.gray)
plt.show()

# For 2b)    
def mirror_vertically(image):
    return image[::-1, ...]

def mirror(image, horizontal = True, vertical = False):
    """
    Mirrors an image. Pass boolean parameters to this function.
    Image arrays are multidimensional arrays. When accessing with Python [] as seen in slides, we can access the different
    arrays by using ,.
    So image[x,y] where x is the arrayindex on the vertical axis and y the arrayindex for on the horizontal axis, gets the color array or grey value
    (as seen in slides)

    Python accessors are a:b:c , where a is from index including a, until excluding b, c is the stepping (2 for every second element)
    When a or b are not defined, it means go from start to finish
    For reverting we can use the accessor expression to select a sublist including all elements with reverted (-1) stepping.

    Lists are immutable so we do a shallow copy of the element, we could save the view without overriding the original image.
    Any changes to the original image(the color values, not the ordering) should still be shown in the flipped view.

    If this is not desired we need to make a DEEP COPY of the original object.
    """

    if horizontal:
        image = mirror_horizontally(image)

    if vertical:
        image = mirror_vertically(image)

    return image


m = mirror(l, horizontal=False, vertical=True)
plt.axis('off')
plt.imshow(l, cmap=plt.cm.gray)
plt.show()
plt.axis('off')
plt.imshow(m, cmap=plt.cm.gray)
plt.show()

m = mirror(l, horizontal=True, vertical=False)
plt.axis('off')
plt.imshow(m, cmap=plt.cm.gray)
plt.show()
