#!/usr/bin/env python
# coding: utf-8
import cv2
import datetime
import os
import time
import numpy as np
import glob
import json 


###############################################################################
#                                  Constants                                  #
###############################################################################
WINDOW_NAME = 'CameraDemo'
CALIBRATIONIMAGEPATH = os.path.abspath("CalibrationImage/") + "/"
CAPTUREPATH = os.path.abspath("CaptureImage/") + "/"
RESOLUTION = (1280,720)
WINDOW_WIDTH=1280
WINDOW_HEIGHT=720
FRAME_RATE = 30

###############################################################################
#                                  Fonctions                                  #
###############################################################################
def open_cam_onboard(width, height, resolution, frame_rate):
    """
    Function generating the VideoCapture with the arguments corresponding to the
    onboard camera
    """
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int){}, height=(int){}, '
               'format=(string)I420, framerate=(fraction){}/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(resolution[0],resolution[1],frame_rate,width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_window(width, height):
    """
    Window handling.
    width,height being the with  and height of the windows
    """
    #Name
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #Size
    cv2.resizeWindow(WINDOW_NAME, width, height)
    #Position
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    #Title
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2')

def create_directory(path):
    """
    Creating a directory at the corresponding path
    """
    try:
        os.mkdir(path)
    except OSError:
        raise OSError("Creation of the directory %s failed" % path)


###############################################################################
#                                  Main                                       #
###############################################################################

if __name__ == "__main__":
    #Openning camera
    cap = open_cam_onboard(WINDOW_WIDTH,WINDOW_HEIGHT,RESOLUTION,FRAME_RATE)
    #Creating windows
    open_window(WINDOW_WIDTH,WINDOW_HEIGHT)

    # Size of the image A4 paper
    ChessboardSize = (21,27)
    # Number of vertices searched in the image
    patternDimension = (7,6)
    # In order to do a fast check before the final check
    chessboardResearchFlag = cv2.CALIB_CB_FAST_CHECK


    # Half the side length of the search window.
    winSize = (ChessboardSize[0]//2,ChessboardSize[1]//2)
    # Setting -1 ,-1  to not use this parameter
    zeroZone = (-1,-1)
    # Terminaison criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    start_calibrate = False
    # Main loop
    while(True):
        ret_calib = False


        # Capture of each frame
        ret_read, frame = cap.read()

        # Operations on frames
        if ret_read is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calibration has started
            if start_calibrate is True:
                # Find the chessboard corners
                ret_calib, corners = cv2.findChessboardCorners(gray, patternDimension, chessboardResearchFlag)

                if ret_calib is False:
                    # Display the resulting frame
                    cv2.imshow(WINDOW_NAME,gray)
                else:
                    print("Valid Image for calibration has been found :")
                    # We refine the corner location (find the sub-pixel accurate location of corners)
                    corners2 = cv2.cornerSubPix(gray,corners, winSize, zeroZone, criteria)

                    #Coping frame to display it.
                    #We want to frame : one that will be write (gray) and one that will be display (img)
                    img = gray.copy()
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, patternDimension, corners2, ret_calib)
                    cv2.imshow(WINDOW_NAME, img)

                    print("Do you want to keep this Image for Calibration ? y/n")
                    # Handling user response
                    while(True):
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord("y"):
                            # Writing the images
                            # Creating the directory if it doesn't exit
                            if not os.path.isdir(CALIBRATIONIMAGEPATH):
                                create_directory(CALIBRATIONIMAGEPATH)

                            path = CALIBRATIONIMAGEPATH + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".png"
                            result = cv2.imwrite(path,gray)
                            if result is False:
                                print("Fail to Save the Image.\n")
                            else:
                                print("Image saved in CalibrationImage Folder.\n")
                            break
                        # Not doing anything
                        elif key == ord("n"):
                            break

                    start_calibrate = False

            else:
                # Otherwise you juste show the gray frame
                cv2.imshow(WINDOW_NAME,gray)

        key = cv2.waitKey(1) & 0xFF
        # Main user input reading
        if key == ord("r"):
            # Just saving a single image
            # Creating the directory if it doesn't exit
            if not os.path.isdir(CAPTUREPATH):
                create_directory(CAPTUREPATH)

            path = CAPTUREPATH + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".png"
            result = cv2.imwrite(path,gray)
            if result is False:
                print("Fail to Save the Image.\n")
            else:
                print("Image saved in CalibrationImage Folder.\n")
            break
        elif key == ord("q"):
            # Quit
            break
        elif key == ord("t"):
            # Start calibration
            start_calibrate = True
    # End of the main loop; we close the windows and the camera
    cv2.destroyAllWindows()
    cap.release()

    # Variables initialisation
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    objectPoints = []
    imagePoints = []

    #Reading all files.
    images = glob.glob(CALIBRATIONIMAGEPATH + "*.png")

    if(len(images) > 0):
        #If we recorded 0 images
        nb_pattern_found = 0
        for image in images:
            # Load image file as a grayscale frame
            img = cv2.imread(image, 0)

            # Find the chessboard corners of a 10x7 Chessboard
            ret_calib, corners = cv2.findChessboardCorners(img, patternDimension, None)

            if ret_calib is False:
                print("Error : No Pattern found in " + image + ".")
            else:
                objectPoints.append(objp)

                # We refine the corner location
                corners2 = cv2.cornerSubPix(img, corners, winSize, zeroZone, criteria)

                # Images Points lists
                imagePoints.append(corners2)
                nb_pattern_found = nb_pattern_found + 1

        if nb_pattern_found > 0:
            # We found some pattern, we start the camera calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints,imagePoints, gray.shape[::-1], None, None)

            if ret is False:
                print("Camera Calibration Failed : cv2.calibrateCamera failed unexpectedly.")
            else:
                print("Camera Matrix : ",mtx)
                mtx = mtx.tolist()
                print("Distorsion Coefficients : ",dist)
                dist = dist.tolist()
                print("Rotation Vectors : ", rvecs)
                print("Translation Vectors : ", tvecs)

                # Json formattation
                tmp = []
                for n in tvecs:
                    tmp1 = []
                    for m in n:
                        tmp1.append(list(m))
                    tmp.append(list(tmp1))

                tvecs = list(tmp)

                tmp = []
                for n in rvecs:
                    tmp1 = []
                    for m in n:
                        tmp1.append(list(m))
                    tmp.append(list(tmp1))

                rvecs = list(tmp)

                data = {"Camera Matrix":mtx, "Distorsion Coefficients":dist, "Rotation Vectors":rvecs, "Translation Vectors":tvecs}
                #Json writing
                with open('CameraCalibrationResult.json', 'w') as outfile:
                    json.dump(data, outfile)

        else:
            print("Camera Calibration Failed : No enough patterns found on images.")
    else:
        print("Camera Calibration Failed : No enough Images.")
