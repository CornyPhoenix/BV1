# -*- coding: utf-8 -*-
# *****************************************************************************
#Module:    Bildverarbeitung 1
#
#Exercise:  05 Image Compression and Segmentation
#
#Authors:   Brand,      Axel       (6145101)
#           Elfaramawy, Nourhan    (6517858)
#           Toprak,     Sibel      (6712316)
# *****************************************************************************


# PRACTICAL PROBLEMS - (Lossy) Image Compression 

# (-> Task 2.1 a)

import numpy as np
from numpy import linalg as la

# Given covariance matrix
covariance_matrix = np.matrix([ [15, 5,  9,  3], \
                                [5,  15, 3,  9], \
                                [9,  3,  15, 5], \
                                [3,  9,  5, 15]]) / 4

# Computed eigenvalues and eigenvectors.
# The i-th column of eigenvectors corresponds to i-th eigenvalue.
eigenvalues, eigenvectors = la.eig(covariance_matrix)

# Sorting the eigenvalues in decreasing order and the eigenvalues accordingly.
# Apparently, la.eig does not necessarily return them ordered that way.
indices = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[indices]
sorted_eigenvectors = eigenvectors[:, indices]

# Transformation matrix A_3 consisting of the eigenvectors corresponding to the
# three largest eigenvalues.
A_3 = sorted_eigenvectors[:, 0:3].T

# Print results
print "eigvals ="
print sorted_eigenvalues
print "eigvecs ="
print sorted_eigenvectors
print "A_3 ="
print A_3

'''
import numpy as np
from numpy import linalg as la

x = np.matrix([[15,5,9,3],[5,15,3,9],[9,3,15,5],[3,9,5,15]])/4

ew, ev = la.eig(x)

y = np.dot(ev[0:3],x)

x2 = np.dot(ev[0:3].T,y)

print x 
print y
print x2
'''

# (-> Task 2.1 b)

print A_3.T*A_3
# Take the first line:
# [0.75 0.25 0.25 -0.25]