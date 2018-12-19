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


    #prec = cv2.adaptiveThreshold(prec,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #curr = cv2.adaptiveThreshold(curr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #ret3,prec = cv2.threshold(prec,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret3,curr = cv2.threshold(curr,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #prec = cv2.Canny(prec,95,190)
    #curr = cv2.Canny(curr,95,190)

    # Compute Difference between two consecutive frame and take the one appearing on the current frame
    #added = cv2.add(curr, prec)
    #curr_without_back = cv2.absdiff(added, prec)
    #frame_delta = cv2.bitwise_and(curr_without_back, curr)

    thresh = 70

    tmp = prec - curr
    frame_delta = cv2.absdiff(curr, tmp)
    frame_delta = cv2.bitwise_and(frame_delta, curr)
    frame_delta[frame_delta<thresh] = 0
    #frame_delta = cv2.absdiff(frame_delta, prec)
    #frame_delta = cv2.bitwise_and(frame_delta, curr)

    #match = cv2.matchTemplate(overlap2, overlap1, cv2.TM_SQDIFF)

    #frame_delta = cv2.bitwise_and(frame_delta, overlap2)
    #frame_delta = cv2.subtract(overlap2, overlap1)
    #frame_delta = cv2.fastNlMeansDenoising(frame_delta)


    open_window("frame_delta")
    cv2.imshow('frame_delta',frame_delta)

    return mask_motion_detection(frame_delta,to_disp)

def mask_motion_detection(frame,to_disp):
    #Create a threshold to exclude minute movements
    thresh = cv2.threshold(frame,25,255,cv2.THRESH_BINARY)[1]

    #Dialate threshold to further reduce error
    thresh = cv2.dilate(thresh,None,iterations=2)

    open_window("Detection Frame")
    cv2.imshow("Detection Frame",thresh)

    mask = np.zeros_like(thresh)

    #Check for contours in our threshold
    _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    min_width = 20
    min_height = 20

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

def bad_motion_detection(fgbg, frame,to_disp):
    #apply background substraction
    fgmask = fgbg.apply(frame)

    #Create a threshold to exclude minute movements
    thresh = cv2.threshold(fgmask,25,255,cv2.THRESH_BINARY)[1]

    #Dialate threshold to further reduce error
    thresh = cv2.dilate(thresh,None,iterations=2)

    #Check for contours in our threshold
    _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour
    for i in range(len(cnts)):
        # If the contour is big enough
        if cv2.contourArea(cnts[i]) > 1000:
            # Create a bounding box for our contour
            (x,y,w,h) = cv2.boundingRect(cnts[i])
            # Convert from float to int, and scale up our boudning box
            (x,y,w,h) = (int(x),int(y),int(w),int(h))
            # Initialize tracker
            bbox = (x,y,w,h)

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(to_disp,p1,p2,(0,0,255),5)

    return fgmask

def get_mask(matches,key_points1,key_points2):
    # Minimum number of matching
    MIN_MATCH = 10
    # Parameter to stop computing. Terminaison condition
    MAX_RANSAC_REPROJ_ERROR = 5.0

    if len(matches)>= MIN_MATCH:
        src_pts = list()
        dst_pts = list()

        for m in matches:
            src_pts.append(key_points1[m.queryIdx].pt)
            dst_pts.append(key_points2[m.trainIdx].pt)

        src_pts = np.float32([src_pts]).reshape(-1,1,2)
        dst_pts = np.float32([dst_pts]).reshape(-1,1,2)


        #Creating the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,MAX_RANSAC_REPROJ_ERROR)

        return mask
    else:
        print("Not enough matches are found : " + str(len(matches)) + " < " + str(MIN_MATCH) + ".")
        return None