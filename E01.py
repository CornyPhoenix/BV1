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


def mirror(image, horizontal, vertical):
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

    If this is not desired we need to make a DEEP COPY of the original object-
    """

    if horizontal:
        image = image[..., ::-1]

    if vertical:
        image = image[::-1, ...]

    return image

# Example image from scipy
l = misc.lena()
m = mirror(l, horizontal=True, vertical=False)
plt.imshow(m, cmap=plt.cm.gray)
# needed in intelliJ to show plots and in spyder if we want to display more than one image plot
plt.show()
