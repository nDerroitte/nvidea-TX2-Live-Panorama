import cv2
import numpy as np
import imutils

from Panorama import *

RESOLUTION = (1280,720)

def get_overlapping_parts(frame1,frame2, proj_matrix):
    warp1, warp2, translation = get_transfo(frame1,frame2, proj_matrix)

    overlap1 = None
    overlap2 = None

    if(translation is not None):

        if(warp1.shape[1] > warp2.shape[1]):
            overlap_length = warp2.shape[1]
        else:
            overlap_length = warp1.shape[1]


        if(translation > 0):
            overlap1 = warp1[int(translation) : overlap_length,:]
            overlap2 = warp2[0:overlap_length - int(translation), :]
        else:
            translation = abs(translation)
            overlap2 = warp2[int(translation) : overlap_length, :]
            overlap1 = warp1[0:overlap_length - int(translation), :]

        overlap1 = get_cartesian(cv2.resize(overlap1, RESOLUTION), proj_matrix)
        overlap2 = get_cartesian(cv2.resize(overlap2, RESOLUTION), proj_matrix)

    else:
        print("Error : No transformation found between 2 frames")

    return overlap1, overlap2

def motion_detection(fgbg,prec_gray, gray, to_disp, proj_matrix):


    # Get the overlapping part of both consecutive frame (only common part is relevant for motion detection
    overlap1, overlap2 = get_overlapping_parts(prec_gray, gray, proj_matrix)

    if(overlap1 is None or overlap2 is None):
        print("No overlapping found")
        return

    # Blur footage to prevent artifacts
    overlap1= cv2.GaussianBlur(overlap1,(21, 21),0)
    overlap2 = cv2.GaussianBlur(overlap2,(21, 21),0)

    # Compute Difference between two consecutive frame
    frame_delta = cv2.absdiff(overlap2, overlap1)
    frame_delta = cv2.bitwise_and(frame_delta, overlap2)
    #frame_delta = cv2.fastNlMeansDenoising(frame_delta)

    open_window("frame_delta")
    cv2.imshow('frame_delta',frame_delta)

    return mask_motion_detection(frame_delta,to_disp)

    '''# params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    p0 = cv2.goodFeaturesToTrack(prec_gray, mask = None, **feature_params)

    flow = cv2.calcOpticalFlowFarneback(overlap1,overlap2, p0, 0.5, 3, 15, 3, 5, 1.2, 0)

    #Project the coordinates into the polar plane
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv = np.zeros((RESOLUTION[1],RESOLUTION[0],3), np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)


    open_window("HSV")
    cv2.imshow("HSV", hsv)
    return None'''

def mask_motion_detection(frame,to_disp):
    #Create a threshold to exclude minute movements
    thresh = cv2.threshold(frame,25,255,cv2.THRESH_BINARY)[1]

    #Dialate threshold to further reduce error
    thresh = cv2.dilate(thresh,None,iterations=2)

    mask = np.zeros_like(thresh)

    #Check for contours in our threshold
    _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour
    for i in range(len(cnts)):
        #tmp = np.zeros(thresh.shape,np.uint8)
        #cv2.drawContours(tmp,[cnts[i]],0,255,-1)
        # If the contour is big enough (and dense enough to represent an object).
        if cv2.contourArea(cnts[i]) > 1000: #and cv2.mean(thresh,mask = tmp)[0] > 120:
            # Create a bounding box for our contour
            (x,y,w,h) = cv2.boundingRect(cnts[i])
            # Convert from float to int, and scale up our boudning box
            (x,y,w,h) = (int(x),int(y),int(w),int(h))
            # Initialize tracker
            bbox = (x,y,w,h)

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(to_disp,p1,p2,(0,0,255),5)
            cv2.fillPoly(mask, pts =[np.array(cnts[i])], color=(255,255,255))
            #cv2.drawContours(mask, cnts, i, 255, 3)
            print("Motion Detected")

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
