import cv2
import numpy as np
from feature import *

###############################################################################
#                                  Fonctions                                  #
###############################################################################
def get_homography_matrix(img1, img2, display = False):
    """
    Computing the homography matrix between 2 images.
    """
    # Finding the matches between the 2 images
    matches, kp1, kp2 = match_images(img1, img2, display)
    # Main function to get homography matrix
    homo_matrix = get_transfo("homography",matches,kp1,kp2)

    return homo_matrix

def get_decomposed_homo_matrix(homo_matrix, cam_matrix):
    """
    Extracting the rotation matrix, transformation matrix (unused), and normals (unused)
    """
    retval, rotation_matrix,trans_matrix, normals = cv2.decomposeHomographyMat(homo_matrix, cam_matrix)

    return retval, rotation_matrix,trans_matrix, normals

def get_affine_transfo(img1, img2, display = False):
    """
    Computing the affine transformation between 2 images.
    """
    # Finding the matches between the 2 images
    matches, kp1, kp2 = match_images(img1, img2, display)
    # Main function to get the affine transformation
    affine_transfo = get_transfo("affine",matches,kp1,kp2)

    return affine_transfo

def get_transfo(method,matches,key_points1,key_points2):
    # Minimum number of matching
    MIN_MATCH = 10
    # Parameter to stop computing. Terminaison condition
    MAX_RANSAC_REPROJ_ERROR = 5.0

    if len(matches)>MIN_MATCH:
        src_pts = list()
        dst_pts = list()

        for m in matches:
            src_pts.append(key_points1[m.queryIdx].pt)
            dst_pts.append(key_points2[m.trainIdx].pt)

        src_pts = np.float32([src_pts]).reshape(-1,1,2)
        dst_pts = np.float32([dst_pts]).reshape(-1,1,2)

        if(method == "homography"):
            #Creating the homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,MAX_RANSAC_REPROJ_ERROR)
        elif(method == "affine"):
            #Creating the affine transformation
            M = cv2.estimateRigidTransform(src_pts, dst_pts,False)
        else:
            print("Error: Unimplemented method " + method + " for function \"get_transfo\"" )
            exit(-1)

        return M
    else:
        print("Not enough matches are found : " + str(len(matches)) + " < " + str(MIN_MATCH) + ".")
        return None
