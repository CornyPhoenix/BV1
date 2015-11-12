# -*- coding: utf-8 -*-
# *****************************************************************************
#Module:    Bildverarbeitung 1
#
#Exercise:  03 Perspective Projections
#
#Authors:   Toprak,     Sibel      (6712316)
#           Elfaramawy, Nourhan    (6517858)
#           Brand,      Axel       (6145101)
# *****************************************************************************

import numpy as np
import math

'''
PART 1:
Describing the given scene point with respect to the camera coordinate system:
'''
#Step 1: Displacement
def displace(x, y, z, vector):
    return np.matrix([[1, 0, 0, -x], 
                      [0, 1, 0, -y], \
                      [0, 0, 1, -z], \
                      [0, 0, 0, 1]]) \
                      * vector

#Step 2.1: Rotation around y
def pan(angle, vector):
    radian = math.radians(angle)
    sin, cos = math.sin(radian), math.cos(radian) 

    return np.matrix([[cos, 0, -sin, 0],
                     [0, 1, 0, 0], \
                     [sin, 0, cos, 0], \
                     [0, 0, 0, 1]]) \
                     * vector
                     
#Step 1.2: Rotation around x
def nick(angle, vector):
    radian = math.radians(angle)
    sin, cos = math.sin(radian), math.cos(radian) 
    
    return np.matrix([[1, 0, 0, 0],
                      [0, cos, sin, 0], \
                      [0, -sin, cos, 0], \
                      [0, 0, 0, 1]]) \
                      * vector                    

'''
PART 2:
Performing the perspective projection:
'''
#Step 3:
def project(focal_length, vector):
    z = vector.item(2)
    factor = focal_length / z
    return np.matrix([[factor, 0, 0, 0], 
                      [0, factor, 0, 0], \
                      [0, 0, 0, focal_length], \
                      [0, 0, 0, 1]]) \
                      * vector

# *****************************************************************************

'''
Lengths in centimeters, angles in degrees!
'''
scene_point = np.matrix([[100], [75], [150], [1]])
'''
Assuming right-handed coordinate system.
'''
result = project(3.5, nick(60, pan(45, displace(0, 300, 0, scene_point))))
print result 
# [[-0.43688007][ 0.50160233][ 3.5 ][ 1. ]]