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


def show(image):
    '''
    Shows an image in console.
    '''
    plot.axis('off')
    plot.imshow(image, cmap=plot.cm.gray)
    plot.show()
    

def custom_fft2(image):
    '''
    Computes 2D Fourier transform using available 1D FFT function.
    '''
    # Get dimensions of the image
    rows, cols = np.shape(image)
    
    # Perform 1D FFT on each row (real to complex)
    row_wise = np.zeros((rows, cols), 'complex')
    for i in range(rows): # 0 ... #rows-1
        row = image[i : i + 1, : ]
        row_wise[i, :] = np.fft.fft(row)
    
    # Perform 1D FFT on each column (complex to complex)
    col_wise = np.zeros((rows, cols), 'complex')
    for j in range(cols): # 0 ... #cols-1
        col = row_wise[ : , j : j + 1].T
        col_wise[:, j] = np.fft.fft(col).T[:, 0]
    
    return col_wise
    
def custom_ifft2(fft2):
    '''
    Computes 2D inverse Fourier transform using available 1D inverse FFT
    function.
    '''
    # Get dimensions of the 2D Fourier transform
    rows, cols = np.shape(fft2)
    
    # Perform 1D iFFT on each row
    row_wise = np.zeros((rows, cols), 'complex')
    for i in range(rows):
        row = fft2[i : i + 1, : ]
        row_wise[i, :] = np.fft.ifft(row)
    
    # Perform 1D iFFT on each column
    col_wise = np.zeros((rows, cols), 'complex')
    for j in range(cols):
        col = row_wise[ : , j : j + 1].T
        col_wise[:, j] = np.fft.ifft(col).T[:, 0]
    
    return col_wise
    
    
def test_custom_fft2(image):
    '''
    Tests whether the custom implementation of the 2D Fourier transform works
    correctly.
    '''
    # What the available numpy.fft.fft2 returns
    result_expected = np.fft.fft2(image)

    # What our custom implementation actually computes
    result_actual = custom_fft2(image)
    
    # Checking if actual result matches with expected result
    if np.array_equal(result_actual, result_expected):
        print "custom_fft2 is working! :)"
    else:
        print "custom_fft2 not working... :("
    
    
def test_custom_ifft2(image):
    '''
    Tests whether the custom implementation of the 2D inverse Fourier transform
    works correctly.
    '''
    fft2 = np.fft.fft2(image)
    
    # What the available numpy.fft.ifft2 returns
    result_expected = np.fft.ifft2(fft2)
    
     # What our custom implementation actually computes
    result_actual = custom_ifft2(fft2)
    
    # Checking if actual result matches with expected result
    if np.array_equal(result_actual, result_expected):
        print "custom_ifft2 is working! :)"
    else:
        print "custom_ifft2 not working... :("
        
    # Print images
    print "Result obtained with numpy.fft.ifft2: "
    show(result_expected.real)
    print "-----"
    print "Result obtained with custom_ifft2: "
    show(result_actual.real)
    

if __name__ == '__main__':
    
    # Loading and showing image, using lena.png if none specified
    if len(sys.argv) > 1:
        image = misc.imread(sys.argv[1], flatten=True)
    else:
        image = misc.lena()
    
    # Perform tests
    test_custom_fft2(image)
    test_custom_ifft2(image)
    
    
        
    

