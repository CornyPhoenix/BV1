# -*- coding: utf-8 -*-
# *****************************************************************************
#Module:    Bildverarbeitung 1
#
#Exercise:  04 Histograms, Filters, Convolution and Fourier-Transform
#
#Authors:   Brand,      Axel       (6145101)
#           Elfaramawy, Nourhan    (6517858)
#           Toprak,     Sibel      (6712316)
# *****************************************************************************

# *****
# GREYVALUE NORMALIZATION
# *****

import sys
import numpy
from scipy import misc
import matplotlib.pyplot as plot

'''
Showing an image in console
'''
def show(image):
    plot.axis('off')
    plot.imshow(image, cmap=plot.cm.gray)
    plot.show() 

'''
Performing greyscale transformation on input image
'''
def transform_greyscale(image):
    '''Computing pixel value above darkest and below brightest percentile'''
    old_min = image.min()
    old_max = image.max()
    new_min = numpy.percentile(image, 5)
    new_max = numpy.percentile(image, 90)
    
    print "Minimum pixel value:           ", old_min
    print "Maximum pixel value:           ", old_max
    print "Pixel value > 5th percentile:  ", new_min
    print "Pixel value < 90th percentile: ", new_max
    
    '''Applying greyscale transformation to image'''
    (rows, cols) = image.shape
    
    scaling_factor = 255 / (new_max - new_min)
    
    for i in range(rows):
        for j in range(cols):
            new_value_1 = image[i, j] - new_min
            if new_value_1 >= 0:
                image[i, j] = new_value_1
            else:
                image[i, j] = 0
            new_value_2 = image[i, j] * scaling_factor
            if new_value_2 <= 255:
                image[i, j] = new_value_2
            else:
                image[i, j] = 255
    
    print "min: ", image.min()
    print "max: ", image.max()
    print image
    return image         
 
if __name__ == "__main__":
    
    '''Loading and showing original image, using lena.png if none specified'''
    if len(sys.argv) > 1:
        original = misc.imread(sys.argv[1], flatten=True)
    else:
        original = misc.lena()
    
    
    show(original)
    #misc.imsave('original.png', original)
    
    '''Showing and saving output of transformation'''
    output = transform_greyscale(original)
    show(output)
    misc.imsave('output.png', output)