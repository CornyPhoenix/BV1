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
    #plot.imshow(image, cmap=plot.cm.gray)
    plot.hist(image)
    plot.show() 

'''
Performing linear greyscale transformation on input image to change the range 
of the pixel intensity values in that image
'''
def transform_greyscale(image):
    '''Computing pixel value above darkest and below brightest percentile'''
    old_min = image.min()
    old_max = image.max()
    new_min = numpy.percentile(image, 5)
    new_max = numpy.percentile(image, 90)
    
    print "Min. intensity:  ", old_min
    print "Max. intensity:  ", old_max
    print "5th percentile:  ", new_min
    print "90th percentile: ", new_max
    
    (rows, cols) = image.shape
    
    '''Factor by which the histogram is stretched'''
    factor = 255 / (new_max - new_min)
    
    for i in range(rows):
        for j in range(cols):
            
            '''
            The whole histogram is moved along the x-axis (gray-value intensi-
            ty), such that its desired lower bound of the range (5th percen-
            tile) goes towards 0. Those pixels that get a negative value, are
            mapped to 0.
            '''
            intensity_after_translation = image[i, j] - new_min
            if intensity_after_translation >= 0:
                image[i, j] = intensity_after_translation
            else:
                image[i, j] = 0
                
            '''
            After that, the histogram is stretched along the x-axis, such that 
            the desired upper bound of the range (90th percentile) moves to 
            where its old upper bound (255) was.
            '''
            intensity_after_stretching = image[i, j] * factor
            if intensity_after_stretching <= 255:
                image[i, j] = intensity_after_stretching
            else:
                image[i, j] = 255
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