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
# FOURIER-TRANSFORM
# *****

import sys
import numpy as np
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
Custom implemetation of 2D-Fourier-Transform
'''
def custom_fft2(image):
    # Get dimensions of the image
    rows, cols = np.shape(image)
    
    # Do 1D FFT on each row (real to complex)
    row_wise = np.fft.fft(image[0 : 1, : ])
    for i in range(1, rows):
        row = np.fft.fft(image[i : i + 1, : ])
        row_wise = np.concatenate((row_wise, row), 0)
   
    #Do 1D FFT on each column resulting from first step (complex to complex)
    col_wise = np.fft.fft(row_wise[ : , 0 : 1])
    for j in range(1, cols):
        col = np.fft.fft(row_wise[ : , j : j + 1])
        col_wise = np.concatenate((col_wise, col), 1)
    
    return col_wise


if __name__ == '__main__':
    
    # Loading and showing image, using lena.png if none specified
    if len(sys.argv) > 1:
        image = misc.imread(sys.argv[1], flatten=True)
    else:
        image = misc.lena()
    
    # What the available 2D Fourier-Transform function in numpy.fft returns
    result_expected = np.fft.fft2(image)

    # What our custom function computes
    result_actual = custom_fft2(image)
    show(np.fft.ifft2(result_actual).real)

