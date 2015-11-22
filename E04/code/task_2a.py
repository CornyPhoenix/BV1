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
    percentile_dark = numpy.percentile(image, 5)
    percentile_bright = numpy.percentile(image, 90)
    print "Pixel value > 5th percentile:  ", percentile_dark
    print "Pixel value < 90th percentile: ", percentile_bright
    
    '''Applying greyscale transformation to image'''
    (rows, cols) = image.shape
    
    for i in range(rows):
        for j in range(cols):
            
            if image[i,j] <= percentile_dark:
                image[i,j] = 0 
                
            if image[i,j] >= percentile_bright:
                image[i,j] = 255
       
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