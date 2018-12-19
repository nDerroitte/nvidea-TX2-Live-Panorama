import sys
import cv2
from Util import *
import numpy as np
import time

np.set_printoptions(threshold=np.nan)

TOL = 5
def readMask(img_number, folder_name, grpNb, Indoor):
    img_path = folder_name + "/seg_"+str(grpNb)+"_"+str(Indoor)+"_"+"{0:0=4d}".format(img_number)+".png"
    return cv2.imread(img_path, 0)

def maskComp(true_mask, generated_mask, tol = False):
    nb_pixel_too_much = 0
    nb_pixel_missing = 0
    total_true_white_pixel = 0
    for x in range(len(true_mask)):
        print("Iteration {} out of 720".format(x), end = '\r')
        for y in range (len(true_mask[0])):
            if true_mask[x][y]:
                total_true_white_pixel +=1
            if true_mask[x][y] == generated_mask[x][y]:
                continue
            elif true_mask[x][y] and not generated_mask[x][y]:
                if tol and isToleratedNp(generated_mask,x,y):
                    continue
                else:
                    nb_pixel_missing +=1
            elif generated_mask[x][y] and not true_mask[x][y]:
                if tol and isToleratedNp(true_mask,x,y):
                    continue
                else:
                    nb_pixel_too_much +=1
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
    for i in range(-TOL,TOL):
        for j in range (-TOL,TOL):
            if x+i >= 720 or x+i <0 or y+i < 0 or y+i >=1280:
                continue
            if mask[x+i][y+i] == 255:
                return True
    return False

def isToleratedNp(mask, x ,y ):
    if x+TOL >= 720 or x-TOL <0 or y-TOL < 0 or y+TOL >=1280:
        return False
    if np.mean(mask[x-TOL : x+TOL, y-TOL : y+TOL]) > 0:
        return True
    return False

'''if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect use of the function. You should specify the id of the image you want to test")
        print("Example: python3 MaskComp.py 346")
        sys.exit()
    start_time = time.time()
    img_nb = int(sys.argv[1])
    folder_name = "AnoIn"
    img =  readMask(img_nb,folder_name, 6 , 1)
    cv2.imwrite("mc1.png",img)
    img2 = (255-img)
    cv2.imwrite("mc2.png",img2)
    print("Mask and inverse mask. ")
    print("With a tolerence of 5 pixels")
    maskComp(img, img2, True)
    print("Without any tolerence")
    maskComp(img, img2, False)
    print("Mask and white image: ")
    img3= np.full((720,1280),255,np.uint8)
    cv2.imwrite("mc3.png",img3)
    print("With a tolerence of 5 pixels")
    maskComp(img, img3, True)
    print("Without any tolerence")
    maskComp(img, img3, False)
    print("Mask and half white image: ")
    img3[:,0:640] = 0
    print("With a tolerence of 5 pixels")
    maskComp(img, img3, True)
    print("Without any tolerence")
    maskComp(img, img3, False)
    print("Mask and mask itself: ")
    print("With a tolerence of 5 pixels")
    maskComp(img, img, True)
    print("Without any tolerence")
    maskComp(img, img, False)
    print("Mask and black image: ")
    print("With a tolerence of 5 pixels")
    img3= np.zeros((720,1280),np.uint8)
    cv2.imwrite("mc4.png",img3)
    maskComp(img, img3, True)
    print("Without any tolerence")
    maskComp(img, img3, False)
    print("-- Computation time : %s seconds --" % (time.time() - start_time))'''
