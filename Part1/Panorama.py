import cv2
import numpy as np
from Transformation import *

###############################################################################
#                                  Fonctions                                  #
###############################################################################
def compute_projection_matrix(cam_matrix, scaling_factor, resolution):
    """
    Function that project that compute the projection matrix of the cylindrical plane
    """
    # Computing focal length
    focal_length = (cam_matrix[0][0], cam_matrix[1][1])

    # Variables initialisation
    w,h = resolution
    x_c = float(w) / 2.0
    y_c = float(h) / 2.0
    proj_mat = np.zeros((w,h), dtype=object)

    # Transforming from cartesian plane to cylindrical.
    for y in range(h):
        for x in range(w):
            x_ = int(scaling_factor * (np.arctan(float(x-x_c)/focal_length[0])) + x_c)
            y_ = int(scaling_factor * (float(y-y_c)/(np.sqrt(np.square(float(x-x_c)) + np.square(focal_length[0])))) + y_c)
            proj_mat[x][y] = (x_,y_)

    return proj_mat

def get_panorama(method,panorama,frame, cam_matrix, scaling_factor, resolution, projection_matrice):
    """
    Function in charge of getting the panorama
    """
    if(method == "cylindrical"):
        # Warping the image : proj
        return cylindricalWarpImages(panorama,frame, cam_matrix, scaling_factor, resolution, projection_matrice)
    else:
        print("Error : Unknown or Unimplemented panorama method " + method + ".")

def get_cylindrical(img, cam_matrix, scaling_factor,resolution, projection_matrice):
    """
    Function that project the image onto a cylinder image using the projection_matrice
    """
    cyl_proj = np.zeros_like(img)
    h,w = img.shape

    for y in range(h):
        for x in range(w):
            x_, y_ = projection_matrice[x,y]
            if(x_ > 0 and x_ < w and y_ > 0 and y_ < h):
                cyl_proj[y_,x_] = img[y,x]

    return cyl_proj

def cylindricalWarpImages(img1,img2,cam_matrix, scaling_factor, resolution, projection_matrice):
    """
    Function in charge of stiching the cylindrical image together
    """
    # Getting the warp version of the image
    warp2 = get_cylindrical(img2, cam_matrix, scaling_factor,resolution, projection_matrice)
    # Affine transformation
    transfo = get_affine_transfo(warp2,img1)

    if transfo is not None:
        transfo[0][0] = 1
        transfo[0][1] = 0
        transfo[1][0] = 0
        transfo[1][1] = 1
        transfo[1][2] = 0

        cyl_warp = cv2.warpAffine(warp2, transfo, (img1.shape[1] + abs(int(transfo[0][2])),img1.shape[0]))

        output = np.zeros_like(cyl_warp)

        a = np.nonzero(img1)
        b = np.nonzero(cyl_warp)

        output[a] = img1[a]
        output[b] = cyl_warp[b]


        output = cv2.fastNlMeansDenoising(output)

        return output, transfo[0][2]
    else:
        print("Error : No Affine Transformation was found between both images.")

        return img1, None
