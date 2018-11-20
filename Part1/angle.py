import cv2
from math import sqrt,atan2,pow,degrees,tan
import numpy as np

from transformation import *

def get_angle(frame1,frame2,cam_matrix, display = False):
    homo_matrix = get_homography_matrix(frame1, frame2, display)
    retval, rotation_matrix,trans_matrix, normals = get_decomposed_homo_matrix(homo_matrix, cam_matrix)
    angles = degrees(get_euler_angle(rotation_matrix[0]))

    return angles

def get_euler_angle(rotation_matrix):
    x_angle = np.arctan2(rotation_matrix[2][1],rotation_matrix[2][2])
    y_angle = np.arctan2(-rotation_matrix[2][0], np.sqrt(np.square(rotation_matrix[2][1])  + np.square(rotation_matrix[2][2])))
    z_angle = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])

    return np.array([x_angle, y_angle, z_angle])
