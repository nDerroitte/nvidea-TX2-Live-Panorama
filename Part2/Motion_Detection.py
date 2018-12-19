import cv2
import numpy as np

from Panorama import *
from Feature import *

RESOLUTION = (1280,720)

def motion_detection(fgbg,kernel, prec_gray, gray,proj_matrix, to_disp=None):
    '''
    This function will detect motion betwen two consecutive frames and will return the corresponding
    mask
    '''
    # Get the overlapping part of both consecutive frame (only common part is relevant for motion detection)
    overlap1, overlap2 = get_overlapping_parts(prec_gray, gray, proj_matrix)

    if(overlap1 is None or overlap2 is None):
        print("No overlapping found")
        return

    # Blur footage to prevent artifacts
    prec = cv2.GaussianBlur(overlap1,(21, 21),0)
    curr = cv2.GaussianBlur(overlap2,(21, 21),0)

    #Extract Foreground Mask
    static_fg_mask = fgbg.apply(prec)
    static_fg_mask = cv2.morphologyEx(static_fg_mask, cv2.MORPH_OPEN, kernel)

    #Extract Background Mask
    stat_bg_mask = cv2.bitwise_not(static_fg_mask)
    stat_bg_mask = cv2.threshold(stat_bg_mask,25,255,cv2.THRESH_BINARY)[1]

    #Extract Background
    stat_bg = cv2.bitwise_and(prec,stat_bg_mask)

    #Extract Foreground Mask
    moving_fg_mask = fgbg.apply(curr)
    moving_fg_mask = cv2.morphologyEx(static_fg_mask, cv2.MORPH_OPEN, kernel)

    #Extract Background Mask
    mov_bg_mask = cv2.bitwise_not(static_fg_mask)
    mov_bg_mask = cv2.threshold(mov_bg_mask,25,255,cv2.THRESH_BINARY)[1]

    #Extract Background
    mov_bg = cv2.bitwise_and(curr,mov_bg_mask)

    # Compute Difference between two consecutive frame and take the elements appearing on the current frame
    frame_delta = cv2.absdiff(mov_bg, stat_bg)
    frame_delta = cv2.bitwise_and(frame_delta,curr)

    return mask_motion_detection(frame_delta,to_disp)

def mask_motion_detection(frame,to_disp):
    """
    This function  will return the mask corresponding to the mask representing the motion
    """
    #Create a threshold to exclude minute movements
    thresh = cv2.threshold(frame,40,255,cv2.THRESH_BINARY)[1]

    #Dialate threshold to further reduce error
    thresh = cv2.dilate(thresh,None,iterations=2)

    mask = np.zeros_like(thresh)

    #Check for contours in our threshold
    _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    min_width = 10
    min_height = 10

    # For each contour
    for i in range(len(cnts)):
        # If the contour is big enough.
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