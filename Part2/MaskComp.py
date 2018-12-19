import sys
import cv2
from Util import *
import numpy as np
import time

###############################################################################
#                                  Constants                                  #
###############################################################################
TOL = 5
###############################################################################
#                                  Functions                                  #
###############################################################################
def readMask(img_number, folder_name, grpNb, Indoor):
    """
    From the image number and the parameter to find to corresponding folder, create a cv2 frame
    """
    img_path = folder_name + "/seg_"+str(grpNb)+"_"+str(Indoor)+"_"+"{0:0=4d}".format(img_number)+".png"
    return cv2.imread(img_path, 0)

def maskComp(true_mask, generated_mask, tol = False):
    """
    Main algorithm responsible for the performance assessment of the motion detection part.
    Return a error from 0 to 100
    """
    #Variable initialisation
    nb_pixel_too_much = 0
    nb_pixel_missing = 0
    total_true_white_pixel = 0
    #Loop on the image
    for x in range(len(true_mask)):
        print("Iteration {} out of 720".format(x), end = '\r')
        for y in range (len(true_mask[0])):
            #If the pixel is white
            if true_mask[x][y]:
                total_true_white_pixel +=1
            #If the two pixels correspond
            if true_mask[x][y] == generated_mask[x][y]:
                continue
            #If they don't
            elif true_mask[x][y] and not generated_mask[x][y]:
                #Handle tolerance
                if tol and isToleratedNp(generated_mask,x,y):
                    continue
                else:
                    nb_pixel_missing +=1
            elif generated_mask[x][y] and not true_mask[x][y]:
                if tol and isToleratedNp(true_mask,x,y):
                    continue
                else:
                    nb_pixel_too_much +=1
    #Error computation (see report)
    e1 = nb_pixel_missing*100/total_true_white_pixel
    e21 = nb_pixel_too_much*100/total_true_white_pixel
    e22 = nb_pixel_too_much*100/((1280*720)-total_true_white_pixel)
    if e21 >100.0:
        e21 = 100.0
    print("RMS error with total image consideration : {}".format(rms((e1,e22))))
    rms_error = rms((e1,e21))
    print("RMS error with shape consideration : {}".format(rms_error))
    return rms_error

def isTolerated(mask,x,y):
    """
    Exhaustif apply of the tolerance. UNUSED.
    """
    for i in range(-TOL,TOL):
        for j in range (-TOL,TOL):
            if x+i >= 720 or x+i <0 or y+i < 0 or y+i >=1280:
                continue
            if mask[x+i][y+i] == 255:
                return True
    return False

def isToleratedNp(mask, x ,y ):
    """
    Tolerance apply using numpy
    """
    if x+TOL >= 720 or x-TOL <0 or y-TOL < 0 or y+TOL >=1280:
        return False
    if np.mean(mask[x-TOL : x+TOL, y-TOL : y+TOL]) > 0:
        return True
    return False