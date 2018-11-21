import cv2
import json
import sys
import operator

import numpy as np

from angle import *
from panorama import *
from jetson_cam import *
from reader import *

PROJECTION_MATRICE = None
IMPLEMENTED_MODE = ["panorama", "matching_demo"]
FRAME_NB_BTW_PANO = 50
#RESOLUTION = (640,480)
RESOLUTION = (1280,720)

def get_cam_matrix(filename):
    json_data = open(filename).read()

    data = json.loads(json_data)
    return np.array(data["Camera Matrix"])

def video_matching_demo(cap,cam_matrix):
    relative_angle = [0.0, 0.0, 0.0]

    if(len(cap) > 0):
        frame = cap.pop()
    else:
        print("Error: 0 frame in the video mentionned.")
        exit(-1)

    while(len(cap) > 0):
        prec_frame = frame
        frame = cap.pop()
        cv2.resize(frame, RESOLUTION)

        angle = get_angle(prec_frame, frame, cam_matrix, True)
        relative_angle = list(map(operator.add, relative_angle,angle))
        #cv2.putText(panorama, ("x-angle:" + str(relative_angle[0]) + " - y-angle:" + str(relative_angle[1]) + " - z-angle:" + str(relative_angle[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            while(True):
                key = cv2.waitKey(0) & 0xFF
                if(key == ord("p")):
                    break
        elif key == ord("r"):
            frame = None
            relative_angle = [0.0, 0.0, 0.0]
            cap = frameReadingFromImage(video_dirname)

    cv2.destroyAllWindows()

def video_panorama(cap,cam_matrix, video_dirname):
    relative_angle = [0.0, 0.0, 0.0]
    panorama = None
    nb_frame = FRAME_NB_BTW_PANO

    scaling_factor = cam_matrix[0][0] #Scaling Factor equal to focal length

    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    rec_pos = (0,0)

    if(len(cap) > 0):
        frame = cap.pop()
    else:
        print("Error: 0 frame in the video mentionned.")
        exit(-1)

    while(len(cap) > 0):
        prec_frame = frame
        frame = cap.pop()
        cv2.resize(frame, RESOLUTION)

        if panorama is None:
            panorama = get_cylindrical(frame, cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)

        angle = get_angle(prec_frame, frame, cam_matrix)

        if(nb_frame > 0):
            nb_frame = nb_frame - 1;
        else:
            panorama, translation = get_panorama("cylindrical",panorama,frame.copy(), cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)
            nb_frame = FRAME_NB_BTW_PANO

            panorama_to_display = cv2.cvtColor(panorama.copy(), cv2.COLOR_GRAY2BGR)

            rec_pos = (int(translation),0)
            cv2.rectangle(panorama_to_display,rec_pos,(RESOLUTION[0]+rec_pos[0],RESOLUTION[1] + rec_pos[1]),(0,0,255),10)

            open_window("Panorama")
            cv2.imshow("Panorama", panorama_to_display)
            #cv2.putText(panorama, ("x-angle:" + str(relative_angle[0]) + " - y-angle:" + str(relative_angle[1]) + " - z-angle:" + str(relative_angle[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))
        relative_angle = list(map(operator.add, relative_angle,angle))


        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            while(True):
                key = cv2.waitKey(0) & 0xFF
                if(key == ord("p")):
                    break
        elif key == ord("r"):
            frame = None
            relative_angle = [0.0, 0.0, 0.0]
            panorama = None
            nb_frame = FRAME_NB_BTW_PANO
            cap = frameReadingFromImage(video_dirname)

    if panorama is not None:
        print("Panorama Saved")
        cv2.imwrite(("Panorama_" + str(video_dirname) + ".jpg") ,panorama)
    else:
        print("Error: The panorama has not been computed.")

    cv2.destroyAllWindows()

def live_matching_demo(cap,cam_matrix):
    ret = False
    frame = None

    relative_angle = [0.0, 0.0, 0.0]

    start_live = False

    while(cap.isOpened()):
        prec_ret, prec_frame = (ret,frame)
        ret, frame = cap.read()

        if ret is True:
            cv2.resize(frame, RESOLUTION)

        if panorama is None:
            panorama = get_cylindrical(frame, cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)

        if prec_ret is True and ret is True and start_live is True:
            angle = get_angle(prec_frame, frame, cam_matrix, True)

            relative_angle = list(map(operator.add, relative_angle,angle))
            #cv2.putText(panorama, ("x-angle:" + str(relative_angle[0]) + " - y-angle:" + str(relative_angle[1]) + " - z-angle:" + str(relative_angle[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            while(True):
                key = cv2.waitKey(0) & 0xFF
                if(key == ord("p")):
                    break
        elif key == ord("s"):
            start_live = not start_live
        elif key == ord("r"):
            ret = False
            relative_angle = [0.0, 0.0, 0.0]
            cap.release()
            cap = cv2.VideoCapture(video_filename)

    cap.release()
    cv2.destroyAllWindows()

def live_panorama(cap,cam_matrix, video_dirname):
    ret = False
    frame = None

    relative_angle = [0.0, 0.0, 0.0]
    panorama = None
    nb_frame = FRAME_NB_BTW_PANO

    scaling_factor = cam_matrix[0][0] #Scaling Factor equal to focal length

    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    start_pano = False

    while(cap.isOpened()):
        prec_ret, prec_frame = (ret,frame)
        ret, frame = cap.read()

        if ret is True:
            cv2.resize(frame, RESOLUTION)

        if panorama is None and start_pano is True:
            panorama = get_cylindrical(frame, cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)

        if prec_ret is True and ret is True and start_pano is True:
            angle = get_angle(prec_frame, frame, cam_matrix)
            if(nb_frame > 0):
                nb_frame = nb_frame - 1;
            else:
                panorama = get_panorama("cylindrical",panorama,frame.copy(), cam_matrix, scaling_factor, RESOLUTION, PROJECTION_MATRICE)
                nb_frame = FRAME_NB_BTW_PANO

                open_window("Panorama")
                cv2.imshow("Panorama", panorama)
                #cv2.putText(panorama, ("x-angle:" + str(relative_angle[0]) + " - y-angle:" + str(relative_angle[1]) + " - z-angle:" + str(relative_angle[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))

            relative_angle = list(map(operator.add, relative_angle,angle))

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            if start_pano is True:
                cv2.imwrite(("Panorama_" + str(video_dirname) + ".jpg") ,panorama)
            start_pano = not start_pano

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live = False
    if(len(sys.argv) == 3):
        cmatrix_filename = sys.argv[1]
        cam_matrix = get_cam_matrix(cmatrix_filename)

        mode = sys.argv[2]
        if mode not in IMPLEMENTED_MODE:
            print("Error: Implemented modes are " + str(IMPLEMENTED_MODE) + ".")
            exit(-1)
        live = True
        print("Live Motion_Detection is Not Implemented Yet")
        exit(-1)
        #cap = cv2.VideoCapture(-1)
        #cap = open_cam_onboard(width, height)
    elif(len(sys.argv) == 4):
        cmatrix_filename = sys.argv[1]
        cam_matrix = get_cam_matrix(cmatrix_filename)

        live = False
        mode = sys.argv[2]
        if mode not in IMPLEMENTED_MODE:
            print("Error: Implemented modes are " + str(IMPLEMENTED_MODE) + ".")
            exit(-1)
        video_dirname = sys.argv[3]

        #cap = cv2.VideoCapture(video_filename)
        cap = frameReadingFromImage(video_dirname)

        if cap is None:
            print("Error: Fail to read the Video Files.")
            exit(-1)

    else:
        print("Error: python3.6 main.py cam_matrix_filename.json mode=" + str(IMPLEMENTED_MODE) + " [video_dirname]")
        exit(-1)

    if live is True and mode == "panorama":
        live_panorama(cap,cam_matrix)
    elif live is True and mode == "matching_demo":
        live_matching_demo(cap,cam_matrix)
    elif live is False and mode == "panorama":
        video_panorama(cap,cam_matrix,video_dirname)
    elif live is False and mode == "matching_demo":
        video_matching_demo(cap,cam_matrix)
