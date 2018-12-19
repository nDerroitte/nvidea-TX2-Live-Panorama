import sys
import time
import math
import os
import cv2
import numpy as np
from util import *
###############################################################################
#                                  Constants                                  #
###############################################################################
TOLERENCE = 3
TOL = TOLERENCE

###############################################################################
#                                  Functions                                  #
###############################################################################
def evaluateError(img_nb, path, grp_nb, video_seq,draw= False):
    """
    Main function responsible for the performance assessment of the person detection module.
    Compare the label generated and the one of reference.
    Return a error score from 0% to 100%
    """
    #Variable initialisation
    file_name = path+"box_{}_{}.txt".format(grp_nb,video_seq)
    generated_file = path + "generatedbox_img_{}_{}".format(grp_nb,video_seq)+"_{0:0=4d}.txt".format(img_nb)
    img_name = "img_{}_{}".format(grp_nb,video_seq)+"_{0:0=4d}.jpg".format(img_nb)
    mask_name = "seg_{}_{}".format(grp_nb,video_seq)+"_{0:0=4d}.png".format(img_nb)
    #Getting the rectangles from the textfiles
    true_rects = getRectFromImg(img_name, file_name)
    generated_rects = getRectFromImg(img_name, generated_file)
    error = []
    #display mode
    if draw :
        frame = cv2.imread(path+img_name)
    for i in range(len(true_rects)):
        #If there is some rectangle generated missing
        if i >= len(generated_rects):
            break
        #display mode
        if draw:
            cv2.rectangle(frame,(true_rects[i].top_left.x, true_rects[i].top_left.y),(true_rects[i].bottom_right.x, true_rects[i].bottom_right.y),(0,0,255),1)
            cv2.rectangle(frame,(generated_rects[i].top_left.x, generated_rects[i].top_left.y),(generated_rects[i].bottom_right.x, generated_rects[i].bottom_right.y),(255,0,0),1)
        #Error computation (see report for detail)
        e1 = insideError(true_rects[i],generated_rects[i], path+mask_name)
        e2 = outsideError(true_rects[i],generated_rects[i])
        #rms error
        error.append(rms((e1,e2)))
    for i in range(abs(len(true_rects)-len(generated_rects))):
        error.append(100)
    total_error = me(error)
    print("Total error on image {}: {}".format(img_nb,total_error))
    #display mode
    if draw :
        cv2.imshow("Compare",frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return total_error

def getRectFromImg(img_name, file_name):
    """
    From a textfile and the image name, return a rectangle corresponding to the lie in the text file
    """
    text_file = open(file_name, "r")
    lines = text_file.read().split('\n')
    rect = []
    for line in lines :
        list = line.split(',')
        if img_name == list[0]:
            #If the name corresponds, we create the rectangle
            pt1 = Point(int(list[1]), int(list[2]))
            width = int(list[3])
            height = int(list[4])
            pt2 = Point(pt1.x+ width, pt1.y + height)
            rect.append(Rectangle(pt1, pt2))
        else:
            continue
    text_file.close()
    #if not len(rect) :
        #print("The image {} is not a reference image and cannot be compare!".format(img_name))
    return rect

def drawRedLines(file_name):
    """
    Display function: draw the reference rectangles on the image sequence
    """
    text_file = open(file_name, "r")
    #Reading file
    lines = text_file.read().split('\n')
    img_name = None
    for line in lines :
        list = line.split(',')
        if img_name == None:
            img_name = list[0]
            #Opening frame corresponding to the refernce image
            img = cv2.imread("AnoOut/"+img_name)
        elif img_name != list[0] :
            #Display
            cv2.imshow(img_name,img)
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                exit()
            img_name = list[0]
            img = cv2.imread("AnoOut/"+img_name)

        pt1 = (int(list[1]), int(list[2]))
        width = int(list[3])
        height = int(list[4])
        pt2 = (pt1[0]+ width, pt1[1] + height)
        cv2.rectangle(img,pt1,pt2,(0,0,255),1)

    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text_file.close()

def attenuateError(seg_path, rect):
    """
    Attenuate the error if <50% of the pixel in the rectangle are not important one
    """
    seg = cv2.imread(seg_path, 0)
    #creating the frame for the mask corresponding to the rectangle to assess
    sub_mask = seg[rect.top_left.y : rect.bottom_right.y,rect.top_left.x:rect.bottom_right.x]
    total_pixel = rect.area
    white_pixel = np.count_nonzero(sub_mask)
    p = white_pixel*100/total_pixel
    if p> 50.0:
        #print("There is too much important pixel in the inside rectangle missing. No attenuation!")
        return rect.area
    else :
        return total_pixel - ((total_pixel-white_pixel)/2)



def insideError(true_rect, generated_rect, seg_path):
    """
    Missing pixel error computing. Using area of the rectangle to do so
    """
    #Area init
    a1 = a2 = a3 = a4 = 0
    #Rectangle 1
    if generated_rect.top_left.x > true_rect.top_left.x+TOL and generated_rect.top_left.x<true_rect.bottom_right.x:
        r1 = Rectangle(Point(true_rect.top_left.x, max(generated_rect.top_left.y,true_rect.top_left.y)), Point(generated_rect.top_left.x,min(generated_rect.bottom_right.y,true_rect.bottom_right.y)))
        a1 = attenuateError(seg_path, r1)
    #Rectangle 2
    if generated_rect.bottom_right.y < true_rect.bottom_right.y -TOL and generated_rect.bottom_right.y > true_rect.top_left.y:
        r2 = Rectangle(Point(true_rect.top_left.x, generated_rect.bottom_right.y), Point(true_rect.bottom_right.x,true_rect.bottom_right.y))
        a2 =  attenuateError(seg_path, r2)
    #Rectangle 3
    if generated_rect.top_left.y > true_rect.top_left.y +TOL and generated_rect.top_left.y < true_rect.bottom_right.y:
        r3 = Rectangle(Point(true_rect.top_left.x,true_rect.top_left.y), Point( true_rect.bottom_right.x, generated_rect.top_left.y))
        a3 =  attenuateError(seg_path, r3)
    #Rectangle 4
    if generated_rect.bottom_right.x < true_rect.bottom_right.x -TOL and generated_rect.bottom_right.x > true_rect.top_left.x:
        r4 = Rectangle(Point(generated_rect.bottom_right.x, max(true_rect.top_left.y, generated_rect.top_left.y)), Point(true_rect.bottom_right.x, min(true_rect.bottom_right.y, generated_rect.bottom_right.y)))
        a4 =  attenuateError(seg_path, r4)
    total = a1 + a2 + a3 + a4
    error =  total*100/ true_rect.area
    #print("Inside error : {}".format(error))
    return error

def outsideError(true_rect, generated_rect):
    """
    Pixels in excess error computation
    """
    a1 = a2 = a3 = a4 = 0
    #Rectangle 1
    if generated_rect.top_left.x < true_rect.top_left.x- TOL:
        r1 = Rectangle(Point(generated_rect.top_left.x, max(generated_rect.top_left.y,true_rect.top_left.y)), Point(true_rect.top_left.x,min(generated_rect.bottom_right.y,true_rect.bottom_right.y)))
        a1 = r1.area
    #Rectangle 2
    if generated_rect.bottom_right.y > true_rect.bottom_right.y + TOL:
        r2 = Rectangle(Point(generated_rect.top_left.x, true_rect.bottom_right.y), Point(generated_rect.bottom_right.x,generated_rect.bottom_right.y))
        a2 = r2.area
    #Rectangle 3
    if generated_rect.top_left.y < true_rect.top_left.y - TOL:
        r3 = Rectangle(Point(generated_rect.top_left.x,generated_rect.top_left.y), Point( generated_rect.bottom_right.x, true_rect.top_left.y))
        a3 = r3.area
    #Rectangle 4
    if generated_rect.bottom_right.x > true_rect.bottom_right.x +TOL:
        r4 = Rectangle(Point(true_rect.bottom_right.x, max(true_rect.top_left.y, generated_rect.top_left.y)), Point(generated_rect.bottom_right.x, min(true_rect.bottom_right.y, generated_rect.bottom_right.y)))
        a4 = r4.area
    total = a1 + a2 + a3 + a4
    error =  total*100/ true_rect.area
    if error >100.0:
        error = 100.0
    #print("Outside error : {}".format(error))
    return error

def boxComp(boxes, img_nb, grp_nb, video_seq, path):
    """
    Algo called when trying to assess the performance.
    First step : create a text file containing the boxes formatted as said in the statement
    Second step: error computation
    Third step: delete the temp file created.
    """
    writeInFile( path, boxes,"img_{}_{}".format(grp_nb,video_seq)+"_{0:0=4d}.jpg".format(img_nb))
    e = evaluateError(img_nb, path, grp_nb, video_seq)
    os.remove(path+"generatedbox_img_{}_{}".format(grp_nb,video_seq)+"_{0:0=4d}.txt".format(img_nb))
    return e
