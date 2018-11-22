import cv2
import math
import numpy as np 
from Transformation import *

###############################################################################
#                                  Fonctions                                  #
###############################################################################
def get_angle(img1,img2,cam_matrix, display = False):
    """
    Fonction responsible for the angle computation from 2 images and the camera
    matrix created during the camera calibration
    """
    #Creating the homography matrix. See transformation.py
    homo_matrix = get_homography_matrix(img1, img2, display)
    #Obtain the diffrent rotation, ..., from the homography matrix. See transformation.py
    retval, rotation_matrix,trans_matrix, normals = get_decomposed_homo_matrix(homo_matrix, cam_matrix)
    #Note that trans_matrix and  normals are not used.

    #Transform to degree
    angle = get_degree_angle(get_euler_angle(rotation_matrix[0]))

    return angle

def get_degree_angle(angle):
    """
    Transforming from rad to deg
    """
    return angle*180/math.pi

def get_euler_angle(rotation_matrix):
    """
    Computing the Euler angle from the rotation matrix extracted from the homography
    matrix.
    """
    x_angle = np.arctan2(rotation_matrix[2][1],rotation_matrix[2][2])
    y_angle = np.arctan2(-rotation_matrix[2][0], np.sqrt(np.square(rotation_matrix[2][1])  + np.square(rotation_matrix[2][2])))
    z_angle = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])

    return np.array([x_angle, y_angle, z_angle])
