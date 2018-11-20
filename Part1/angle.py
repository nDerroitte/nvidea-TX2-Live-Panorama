import cv2
import math
import numpy as np

from transformation import *

def get_angle(img1,img2,cam_matrix, display = False):
    homo_matrix = get_homography_matrix(img1, img2, display)

    retval, rotation_matrix,trans_matrix, normals = get_decomposed_homo_matrix(homo_matrix, cam_matrix)

    angle = get_degree_angle(get_euler_angle(rotation_matrix[0]))

    return angle

def get_degree_angle(angle):
    return angle*180/math.pi

def get_euler_angle(rotation_matrix):
    x_angle = np.arctan2(rotation_matrix[2][1],rotation_matrix[2][2])
    y_angle = np.arctan2(-rotation_matrix[2][0], np.sqrt(np.square(rotation_matrix[2][1])  + np.square(rotation_matrix[2][2])))
    z_angle = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])

    return np.array([x_angle, y_angle, z_angle])
