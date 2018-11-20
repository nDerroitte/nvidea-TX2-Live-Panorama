import cv2
import json
import sys
import operator

import numpy as np

from angle import *
from panorama import *
from jetson_cam import *

PROJECTION_MATRICE = None
IMPLEMENTED_MODE = ["panorama", "matching_demo"]
FRAME_NB_BTW_PANO = 10
#RESOLUTION = (640,480)
RESOLUTION = (1280,720)

def get_cam_matrix(filename):
    json_data = open(filename).read()

    data = json.loads(json_data)
    return np.array(data["Camera Matrix"])

if __name__ == "__main__":
    if(len(sys.argv) == 3):
        cmatrix_filename = sys.argv[1]
        cam_matrix = get_cam_matrix(cmatrix_filename)

        mode = sys.argv[2]
        if mode not in IMPLEMENTED_MODE:
            print("Error: Implemented modes are " + str(IMPLEMENTED_MODE) + ".")
            exit(-1)
        print("Live Motion_Detection is Not Implemented Yet")
        exit(-1)
        #cap = cv2.VideoCapture(-1)
        #cap = open_cam_onboard(width, height)
    elif(len(sys.argv) == 4):
        cmatrix_filename = sys.argv[1]
        cam_matrix = get_cam_matrix(cmatrix_filename)

        mode = sys.argv[2]
        if mode not in IMPLEMENTED_MODE:
            print("Error: Implemented modes are " + str(IMPLEMENTED_MODE) + ".")
            exit(-1)
        video_filename = sys.argv[3]

        cap = cv2.VideoCapture(video_filename)
    else:
        print("Error: python3.6 main.py cam_matrix_filename mode=" + str(IMPLEMENTED_MODE) + " [video_filename]")
        exit(-1)

    ret = False
    frame = None

    relative_angle = [0.0, 0.0, 0.0]
    panorama = None
    panorama_angle = 0
    nb_frame = FRAME_NB_BTW_PANO

    scaling_factor = cam_matrix[0][0] #Scaling Factor equal to focal length

    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    while(cap.isOpened()):
        prec_ret, prec_frame = (ret,frame)
        ret, frame = cap.read()

        if ret is True:
            cv2.resize(frame, RESOLUTION)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if panorama is None:
            #panorama = frame
            panorama = get_cylindrical(frame, cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)

        if prec_ret is True and ret is True:


            if(mode == "panorama"):
                angle = get_angle(prec_frame, frame, cam_matrix)
                if(nb_frame > 0):
                    nb_frame = nb_frame - 1;
                else:
                    panorama = get_panorama("cylindrical",panorama,frame.copy(), cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)
                    nb_frame = FRAME_NB_BTW_PANO

                    open_window("Panorama")
                    cv2.imshow("Panorama", panorama)
                    #cv2.putText(panorama, ("x-angle:" + str(relative_angle[0]) + " - y-angle:" + str(relative_angle[1]) + " - z-angle:" + str(relative_angle[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))

            elif(mode == "matching_demo"):
                angle = get_angle(prec_frame, frame, cam_matrix, True)

            relative_angle = list(map(operator.add, relative_angle,angle))



        if(mode == "panorama"):
            key = cv2.waitKey(10) & 0xFF
        elif(mode == "matching_demo"):
            key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            while(True):
                key = cv2.waitKey(0) & 0xFF
                if(key == ord("p")):
                    break
        elif key == ord("r"):
            ret = False
            frame = None
            relative_angle = [0.0, 0.0, 0.0]
            panorama = None
            panorama_angle = 0
            nb_frame = 10
            cap.release()
            cap = cv2.VideoCapture(video_filename)

    cap.release()
    cv2.destroyAllWindows()
