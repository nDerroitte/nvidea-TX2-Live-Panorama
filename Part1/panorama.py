import cv2
import numpy as np

from transformation import *

def compute_projection_matrix(cam_matrix, scaling_factor, resolution):
    focal_length = (cam_matrix[0][0], cam_matrix[1][1])

    w,h = resolution
    x_c = float(w) / 2.0
    y_c = float(h) / 2.0
    proj_mat = np.zeros((w,h), dtype=object)

    for y in range(h):
        for x in range(w):
            x_ = int(scaling_factor * (np.arctan(float(x-x_c)/focal_length[0])) + x_c)
            y_ = int(scaling_factor * (float(y-y_c)/(np.sqrt(np.square(float(x-x_c)) + np.square(focal_length[0])))) + y_c)
            proj_mat[x][y] = (x_,y_)

    return proj_mat

def get_panorama(method,panorama,frame, cam_matrix, scaling_factor, resolution, projection_matrice):
    if(method == "cylindrical"):
        return cylindricalWarpImages(panorama,frame, cam_matrix, scaling_factor, resolution, projection_matrice)
    else:
        print("Error : Unknown or Unimplemented panorama method " + method + ".")

def get_cylindrical(img, cam_matrix, scaling_factor,resolution, projection_matrice):
    #SOURCE = http://pages.cs.wisc.edu/~dyer/cs534/hw/hw4/cylindrical.pdf
    cyl_proj = np.zeros_like(img)
    w,h = resolution

    for y in range(h):
        for x in range(w):
            x_, y_ = projection_matrice[x,y]
            if(x_ > 0 and x_ < w and y_ > 0 and y_ < h):
                cyl_proj[y_,x_] = img[y,x]

    #cyl_proj = cv2.copyMakeBorder(cyl_proj,50,50,300,300, cv2.BORDER_CONSTANT)

    return cyl_proj

def cylindricalWarpImages(img1,img2,cam_matrix, scaling_factor, resolution, projection_matrice):
    warp2 = get_cylindrical(img2, cam_matrix, scaling_factor,resolution, projection_matrice)
    print("lol")
    transfo = get_affine_transfo(warp2,img1)
    print("lol1")

    if transfo is not None:
        transfo[0][0] = 1
        transfo[0][1] = 0
        transfo[1][0] = 0
        transfo[1][1] = 1
        transfo[1][2] = 0

        cyl_warp = cv2.warpAffine(warp2, transfo, (img1.shape[1] + abs(int(transfo[0][2])),img1.shape[0] + abs(int(transfo[1][2]))))
        print("lol2")

        output = np.zeros_like(cyl_warp)
        x,y = output.shape
        x_,y_ = img1.shape

        for i in range(x):
            empty_column = True
            for j in range(y):
                if(i < x_ and j < y_):
                    a = (img1[i][j] == 0).all()
                    b = (cyl_warp[i][j] == 0).all()

                    if a and b:
                        output[i][j] = 0
                    elif a and not b:
                        empty_column = False
                        output[i][j] = cyl_warp[i][j]
                    elif not a and b:
                        empty_column = False
                        output[i][j] = img1[i][j]
                    else:
                        empty_column = False
                        output[i][j] = (img1[i][j])
                        #output[i][j] = (cyl_warp[i][j])
                else:
                    if((cyl_warp[i][j] != 0).all()):
                        empty_column = False
                    output[i][j] = (cyl_warp[i][j])

        print("Frame Stitched")
        return output
    else:
        print("Error : No Affine Transformation was found between both images.")
        return img1
