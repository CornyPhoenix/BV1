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
# CONVOLUTION
# *****

from scipy import misc
from scipy import signal
from numpy.fft import fft2, ifft2

B1 = misc.imread('conveyor_belt.png', flatten=True)
B2 = fft2(B1)**2
misc.imsave('convolution_result.png', ifft2(B2).real)