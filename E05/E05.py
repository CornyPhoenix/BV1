
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
# Image Compression
# *****
import numpy as np
from numpy import linalg as la

x = np.matrix([[15,5,9,3],[5,15,3,9],[9,3,15,5],[3,9,5,15]])/4

ew, ev = la.eig(x)

y = np.dot(ev[0:3],x)

x2 = np.dot(ev[0:3].T,y)

print x 
print y
print x2
