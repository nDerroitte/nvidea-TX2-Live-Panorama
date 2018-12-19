import cv2
import numpy as np

from Panorama import *
from Feature import *

RESOLUTION = (1280,720)


def motion_detection(fgbg,kernel, prec_gray, gray,proj_matrix, to_disp=None):
    # Get the overlapping part of both consecutive frame (only common part is relevant for motion detection)
    overlap1, overlap2 = get_overlapping_parts(prec_gray, gray, proj_matrix)

    if(overlap1 is None or overlap2 is None):
        print("No overlapping found")
        return

    # Blur footage to prevent artifacts
    prec = cv2.GaussianBlur(overlap1,(21, 21),0)
    curr = cv2.GaussianBlur(overlap2,(21, 21),0)

    #prec = fgbg.apply(prec)
    #prec = cv2.morphologyEx(prec, cv2.MORPH_OPEN, kernel)

    #curr = fgbg.apply(curr)
    #curr = cv2.morphologyEx(curr, cv2.MORPH_OPEN, kernel)

    #prec = cv2.adaptiveThreshold(prec,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #curr = cv2.adaptiveThreshold(curr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #ret3,prec = cv2.threshold(prec,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret3,curr = cv2.threshold(curr,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #prec = cv2.Canny(prec,95,190)
    #curr = cv2.Canny(curr,95,190)

    # Compute Difference between two consecutive frame and take the elements appearing on the current frame
    '''frame_delta = cv2.subtract(prec,curr)
    tmp = cv2.absdiff(curr, prec)

    frame_delta = cv2.subtract(tmp,frame_delta)
    frame_delta = cv2.bitwise_and(frame_delta, curr)
    ret,thresh = cv2.threshold(frame_delta,200,255,cv2.THRESH_OTSU)'''

    # combine frame and the image difference
    tmp = cv2.absdiff(curr, prec)
    frame_delta = cv2.addWeighted(prec_gray,0.9,tmp,0.1,0)
    frame_delta2 = cv2.addWeighted(gray,0.9,tmp,0.1,0)

    open_window("frame_delta")
    cv2.imshow('frame_delta',frame_delta)

    return mask_motion_detection(fgbg,kernel,frame_delta,to_disp)

def mask_motion_detection(fgbg,kernel,frame,to_disp):
    #Create a threshold to exclude minute movements
    thresh = cv2.threshold(frame,40,255,cv2.THRESH_BINARY)[1]

    #Dialate threshold to further reduce error
    thresh = cv2.dilate(thresh,None,iterations=2)

    open_window("Detection Frame")
    cv2.imshow("Detection Frame",thresh)

    mask = np.zeros_like(thresh)

    #Check for contours in our threshold
    _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    min_width = 10
    min_height = 10

    # For each contour
    for i in range(len(cnts)):
        # If the contour is big enough (and dense enough to represent an object).
        if cv2.contourArea(cnts[i]) > 1000:
            # Create a bounding box for our contour
            (x,y,w,h) = cv2.boundingRect(cnts[i])

            if(w > min_width and h > min_height):
                # Convert from float to int, and scale up our boudning box
                (x,y,w,h) = (int(x),int(y),int(w),int(h))
                # Initialize tracker
                bbox = (x,y,w,h)

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.fillPoly(mask, pts =[np.array(cnts[i])], color=(255,255,255))
                #cv2.drawContours(mask, cnts, i, 255, 3)

                if(to_disp is not None):
                    cv2.rectangle(to_disp,p1,p2,(0,0,255),5)

    return mask
