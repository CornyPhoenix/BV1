# -*- coding: utf-8 -*-
"""
Module:
=======
Bildverarbeitung 1

Exercise: 
=========
E03 - Image formation

Authors:
========
Toprak,  Sibel      (6712316)
Elfaramawy, Nourhan (6517858)
Brand,   Axel       (6145101)
Möllers, Konstantin (6313136)

"""

from __future__ import division
import colorsys

import numpy as np
import math

from scipy import misc
import matplotlib.pyplot as plt

def sincos(angle):
    rad = math.radians(angle)
    return math.sin(rad), math.cos(rad)

def buildNickMatrix(angle):
    sin, cos = sincos(angle)
    return [
        [1,    0,   0],
        [0,  cos, sin],
        [0, -sin, cos]
    ]
    
def buildPanMatrix(angle):
    sin, cos = sincos(angle)
    return [
        [cos, 0, -sin],
        [  0, 1,    0],
        [sin, 0,  cos]
    ]

class Vector:
    
    def __init__(self, ax, ay, az):
        self.ary = [ax, ay, az]
        
    def __str__(self):
        return "X = %.2fcm; Y = %.2fcm; Z = %.2fcm" % (self.x(), self.y(), self.z())
    
    def x(self):
        """
        Returns the X component of this vector.
        """
        return self.ary[0]
        
    def y(self):
        """
        Returns the Y component of this vector.
        """
        return self.ary[1]
        
    def z(self):
        """
        Returns the Z component of this vector.
        """
        return self.ary[2]
        
    def project(self, focal_length):
        """
        Projects this vector on a plane with a given focal length.
        """
        self.ary = [
            self.x() * focal_length / self.z(),
            self.y() * focal_length / self.z(),
            focal_length
        ]
        
        return self
    
    def translate(self, vec):
        """
        Translates this vector by another.
        """
        self.ary = np.add(self.ary, vec.ary)
        return self
        
    def displace(self, vec):
        """
        Displaces a vector by another vector.
        """
        self.ary = np.subtract(self.ary, vec.ary)
        return self
        
    def nick(self, angle):
        """
        Nicks this vector.
        """
        matrix = buildNickMatrix(angle)
        self.mul(matrix)
        return self
        
    def pan(self, angle):
        """
        Pans this vector.
        """
        matrix = buildPanMatrix(angle)
        self.mul(matrix)
        return self
        
    def mul(self, matrix):
        """
        Multiplies the vector with a matrix and sets the
        resulting coordinates on this vector.
        """
        self.ary = np.dot(matrix, self.ary)
        

# All lengths are given in [cm]!
if __name__ == '__main__':    
    
    # Camera world position
    camera = Vector(0, 300, 0)

    # Table corner world positon
    corner = Vector(100, 75, 150)
    
    # X = -0.44cm; Y = 0.50cm; Z = 3.50cm
    print corner.displace(camera).pan(45).nick(60).project(3.5)
