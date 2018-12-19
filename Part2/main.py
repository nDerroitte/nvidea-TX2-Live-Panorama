import cv2
import json
import sys
import operator

import numpy as np

from Angle import *
from Panorama import *
from JetsonCam import *
from Reader import *
from Motion_Detection import *
from Util import *
from MaskComp import *
from PersonDetection import *

PROJECTION_MATRICE = None
IMPLEMENTED_MODE = ["panorama", "matching_demo", "motion_detection", "enhanced_panorama", "personn_detection"]
FRAME_NB_BTW_PANO = 20
RESOLUTION = (1280,720)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FRAME_RATE = 25

#PERSONN_DETECTION_ALGO = "Opencv"
PERSONN_DETECTION_ALGO = "Tensorflow"

MODEL_PATH = "models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"

def get_cam_matrix(filename):
    json_data = open(filename).read()

    data = json.loads(json_data)
    return np.array(data["Camera Matrix"])

def video_matching_demo(cap,cam_matrix):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    cap2 = cap.copy()
    relative_angle = [0.0, 0.0, 0.0]

    if(len(cap) > 0):
        frame = cap.pop()
    else:
        print("Error: 0 frame in the video mentionned.")
        exit(-1)

    while(len(cap) > 0):
        prec_frame = frame
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)

        angle = get_angle(prec_frame, frame, cam_matrix, True)
        relative_angle = list(map(operator.add, relative_angle,angle))
        cv2.putText(frame, ("angle:" + str(relative_angle[1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("p"):
            while(True):
                key = cv2.waitKey(0) & 0xFF
                if(key == ord("p")):
                    break
        elif key == ord("r"):
            cap = cap2.copy()
            if(len(cap) > 0):
                frame = cap.pop()
            else:
                print("Error: 0 frame in the video mentionned.")
                exit(-1)

            relative_angle = [0.0, 0.0, 0.0]

    cv2.destroyAllWindows()

def video_panorama(cap,cam_matrix):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    init = cap

    focal_length = cam_matrix[0][0]
    scaling_factor = focal_length #Scaling Factor equal to focal length
    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    relative_angle = [0.0, 0.0, 0.0]

    panorama = None
    nb_frame = FRAME_NB_BTW_PANO
    last_frame_in_pano = None
    trans = 0.0

    if(len(cap) < 0):
        print("Error: 0 frame in the video mentionned.")
        exit(-1)

    frame_buffer = list()

    while(len(cap) > 0):
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if panorama is None:
            panorama = get_cylindrical(frame, PROJECTION_MATRICE)
            last_frame_in_pano = frame
            panorama_to_display = cv2.cvtColor(panorama.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(panorama_to_display,(0,0),(RESOLUTION[0],RESOLUTION[1]),(0,0,255),10)
            cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            open_window("Panorama")
            cv2.imshow("Panorama", panorama_to_display)

        if(nb_frame > 0):
            nb_frame = nb_frame - 1;
            if(len(frame_buffer) > FRAME_NB_BTW_PANO - 1):
                del(frame_buffer[0])
            frame_buffer.append(frame)
        else:
            tmp = frame.copy()
            prec_trans = trans

            while(len(frame_buffer) > 0):
                tmp_panorama, tmp_translation = get_panorama("cylindrical",panorama,tmp,last_frame_in_pano,trans,PROJECTION_MATRICE)

                if tmp_translation is None:
                    tmp = frame_buffer.pop()
                else:
                    angle = get_angle(last_frame_in_pano, tmp, cam_matrix)
                    relative_angle = list(map(operator.add, relative_angle,angle))
                    last_frame_in_pano = tmp
                    trans = tmp_translation
                    panorama = tmp_panorama
                    break

            if(len(frame_buffer) < 1):
                print("Error : The panorama can't be made on this Video Sequence (not enough matches could be made).")
                exit(-1)
            elif(len(frame_buffer) > FRAME_NB_BTW_PANO - 1):
                del(frame_buffer[0])
            else:
                FRAME_NB_BTW_PANO = round(FRAME_NB_BTW_PANO/2 + 0.5)
                print("Number of Frame between two panorama has been updated : nb_frame_btw_pano = " + str(FRAME_NB_BTW_PANO))

            frame_buffer.append(frame)
            nb_frame = FRAME_NB_BTW_PANO
            panorama_to_display = cv2.cvtColor(panorama.copy(), cv2.COLOR_GRAY2BGR)


            if(trans > 0):
                cv2.rectangle(panorama_to_display,(int(trans),0),(int(trans) + RESOLUTION[0],RESOLUTION[1]),(0,0,255),10)
                cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), (int(trans),20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            elif(trans < 0 and trans < prec_trans):
                cv2.rectangle(panorama_to_display,(0,0),(RESOLUTION[0],RESOLUTION[1]),(0,0,255),10)
                cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            else:
                cv2.rectangle(panorama_to_display,(int(panorama.shape[1] - (abs(trans) + RESOLUTION[0])),0),(int(panorama.shape[1] - (abs(trans) + RESOLUTION[0]) + RESOLUTION[0]),RESOLUTION[1]),(0,0,255),10)
                cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), ((int(panorama.shape[1] - (abs(trans) + RESOLUTION[0]))), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            open_window("Panorama")
            cv2.imshow("Panorama", panorama_to_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
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
            cap = init

    if panorama is not None:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)
        ret = cv2.imwrite(("Panorama.jpg") ,panorama)
        if ret is False:
            print("Error: Fail to save the Panorama.")
        else:
            print("Panorama Saved")
    else:
        print("Error: The panorama has not been computed.")

    cv2.destroyAllWindows()

def video_motion_detection_demo(cap,cam_matrix):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    cap2 = cap.copy()
    relative_angle = [0.0, 0.0, 0.0]

    focal_length = cam_matrix[0][0]
    scaling_factor = focal_length #Scaling Factor equal to focal length
    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    if(len(cap) > 0):
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)
        #Use grayscale ==> lighter computations
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Blur footage to prevent artifacts
    else:
        print("Error: 0 frame in the video mentionned.")
        exit(-1)

    frame_counter = 50

    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    while(len(cap) > 0):
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)

        if(frame_counter > 0):
            frame_counter = frame_counter - 1
            continue

        #angle = get_angle(prec_frame, frame, cam_matrix, False)
        #relative_angle = list(map(operator.add, relative_angle,angle))

        prec_gray = gray

        #Use grayscale ==> lighter computations
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        to_disp = frame.copy()

        motion_mask = motion_detection(fgbg,kernel, prec_gray, gray, PROJECTION_MATRICE, to_disp)
        #motion_mask = bad_motion_detection(fgbg,frame,to_disp)

        open_window("frame")
        cv2.imshow('frame',to_disp)

        if(motion_mask is not None):
            open_window("Motion Mask")
            cv2.imshow('Motion Mask',motion_mask)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("p"):
            while(True):
                key = cv2.waitKey(0) & 0xFF
                if(key == ord("p")):
                    break
        elif key == ord("r"):
            cap = cap2.copy()
            if(len(cap) > 0):
                frame = cap.pop()
            else:
                print("Error: 0 frame in the video mentionned.")
                exit(-1)

            relative_angle = [0.0, 0.0, 0.0]

    cv2.destroyAllWindows()

def video_enhanced_panorama(cap,cam_matrix):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    init = cap

    focal_length = cam_matrix[0][0]
    scaling_factor = focal_length #Scaling Factor equal to focal length
    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    relative_angle = [0.0, 0.0, 0.0]

    panorama = None
    nb_frame = FRAME_NB_BTW_PANO
    last_frame_in_pano = None
    trans = 0.0

    if(len(cap) < 0):
        print("Error: 0 frame in the video mentionned.")
        exit(-1)

    frame_buffer = list()
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    while(len(cap) > 0):
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if panorama is None:
            panorama = get_cylindrical(frame, PROJECTION_MATRICE)
            last_frame_in_pano = frame
            panorama_to_display = cv2.cvtColor(panorama.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(panorama_to_display,(0,0),(RESOLUTION[0],RESOLUTION[1]),(0,0,255),10)
            cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            open_window("Panorama")
            cv2.imshow("Panorama", panorama_to_display)

        if(nb_frame > 0):
            nb_frame = nb_frame - 1;
            if(len(frame_buffer) > FRAME_NB_BTW_PANO - 1):
                del(frame_buffer[0])
            frame_buffer.append(frame)
        else:
            tmp = frame.copy()
            prec_trans = trans

            while(len(frame_buffer) > 0):
                moving_fg_mask = motion_detection(fgbg, kernel, tmp,last_frame_in_pano,PROJECTION_MATRICE)
                static_fg_mask = compute_foreground_mask(tmp,fgbg,kernel)
                tmp_panorama, tmp_translation = get_enhanced_panorama("cylindrical",panorama,tmp,last_frame_in_pano,trans,PROJECTION_MATRICE, moving_fg_mask, static_fg_mask)

                if tmp_translation is None:
                    tmp = frame_buffer.pop()
                else:
                    angle = get_angle(last_frame_in_pano, tmp, cam_matrix)
                    relative_angle = list(map(operator.add, relative_angle,angle))
                    last_frame_in_pano = tmp
                    trans = tmp_translation
                    panorama = tmp_panorama
                    break

            if(len(frame_buffer) < 1):
                print("Error : The panorama can't be made on this Video Sequence (not enough matches could be made).")
                exit(-1)
            elif(len(frame_buffer) > FRAME_NB_BTW_PANO - 1):
                del(frame_buffer[0])
            else:
                FRAME_NB_BTW_PANO = round(FRAME_NB_BTW_PANO/2 + 0.5)
                print("Number of Frame between two panorama has been updated : nb_frame_btw_pano = " + str(FRAME_NB_BTW_PANO))

            frame_buffer.append(frame)
            nb_frame = FRAME_NB_BTW_PANO
            panorama_to_display = cv2.cvtColor(panorama.copy(), cv2.COLOR_GRAY2BGR)


            if(trans > 0):
                cv2.rectangle(panorama_to_display,(int(trans),0),(int(trans) + RESOLUTION[0],RESOLUTION[1]),(0,0,255),10)
                cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), (int(trans),20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            elif(trans < 0 and trans < prec_trans):
                cv2.rectangle(panorama_to_display,(0,0),(RESOLUTION[0],RESOLUTION[1]),(0,0,255),10)
                cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            else:
                cv2.rectangle(panorama_to_display,(int(panorama.shape[1] - (abs(trans) + RESOLUTION[0])),0),(int(panorama.shape[1] - (abs(trans) + RESOLUTION[0]) + RESOLUTION[0]),RESOLUTION[1]),(0,0,255),10)
                cv2.putText(panorama_to_display, ("angle:" + str(relative_angle[1])), ((int(panorama.shape[1] - (abs(trans) + RESOLUTION[0]))), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))

            open_window("Panorama")
            cv2.imshow("Panorama", panorama_to_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
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
            cap = init

    if panorama is not None:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)
        ret = cv2.imwrite(("Panorama.jpg") ,panorama)
        if ret is False:
            print("Error: Fail to save the Panorama.")
        else:
            print("Panorama Saved")
    else:
        print("Error: The panorama has not been computed.")

    cv2.destroyAllWindows()

def video_personn_detection(video_path):
    global MODEL_PATH

    if(PERSONN_DETECTION_ALGO == "Opencv"):
        detect_opcv(video_path)
    elif(PERSONN_DETECTION_ALGO == "Tensorflow"):
        detect_tf(video_path,MODEL_PATH)
    else:
        print("Error : Unknown Personn Detection algorithm")
        exit(-1)

def motion_detection_assessment(cap, cam_matrix, video_nb):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    focal_length = cam_matrix[0][0]
    scaling_factor = focal_length #Scaling Factor equal to focal length
    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    if(len(cap) > 0):
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)
        #Use grayscale ==> lighter computations
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Blur footage to prevent artifacts
    else:
        print("Error: 0 frame in the video mentionned.")
        exit(-1)


    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    if(video_nb == 1):
        dir_annotation = "Annotation/In/"
        list_id = getRefId(dir_annotation + "box_6_1.txt")
    else:
        dir_annotation = "Annotation/Out/"
        list_id = getRefId(dir_annotation + "box_6_2.txt")


    frame_id = 1
    error_list = list()
    while(len(cap) > 0):
        frame = cap.pop()
        frame = cv2.resize(frame, RESOLUTION)

        prec_gray = gray
        #Use grayscale ==> lighter computations
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        to_disp = frame.copy()

        motion_mask = motion_detection(fgbg,kernel, prec_gray, gray, PROJECTION_MATRICE)

        if(frame_id in list_id):
            print("Image " + str(frame_id) + ":")
            ref_mask = readMask(frame_id,dir_annotation,6,video_nb)
            error_list.append(maskComp(ref_mask,motion_mask,True))

        #motion_mask = bad_motion_detection(fgbg,frame,to_disp)

        frame_id = frame_id + 1

    mean_error = np.mean(np.array(error_list))
    print("Motion Detection Assesment - Mean Error : " + str(mean_error))

    cv2.destroyAllWindows()

def personn_detection_assesment(video_path, video_nb):
    global PERSONN_DETECTION_ALGO

    if(video_nb == 1):
        dir_annotation = "Annotation/In/"
        ann_path = dir_annotation + "box_6_1.txt"
    else:
        dir_annotation = "Annotation/Out/"
        ann_path = dir_annotation + "box_6_2.txt"


    if(PERSONN_DETECTION_ALGO == "Opencv"):
        perf_ass_opcv(video_path,ann_path)
    elif(PERSONN_DETECTION_ALGO == "Tensorflow"):
        perf_ass_tf(video_path,ann_path)
    else:
        print("Error : Unknown Personn Detection algorithm")
        exit(-1)

def live_matching_demo(cap,cam_matrix):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    ret = False
    frame = None

    relative_angle = [0.0, 0.0, 0.0]

    start_live = False

    while(cap.isOpened()):
        prec_ret, prec_frame = (ret,frame)
        ret, frame = cap.read()

        if ret is True:
            frame = cv2.resize(frame, RESOLUTION)

        if prec_ret is True and ret is True and start_live:
            angle = get_angle(prec_frame, frame, cam_matrix, start_live)
            relative_angle = list(map(operator.add, relative_angle,angle))
            cv2.putText(frame, ("angle:" + str(relative_angle[1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))

        if ret is True:
            open_window("Live")
            cv2.imshow("Live", frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("s"):
            start_live = not start_live
            if start_live is False:
                cv2.destroyWindow("Feature Matcher - orb Flanner")
            else:
                relative_angle = [0.0, 0.0, 0.0]

    cap.release()
    cv2.destroyAllWindows()

def live_panorama(cap,cam_matrix):
    global PROJECTION_MATRICE
    global FRAME_NB_BTW_PANO

    frame_buffer = list()
    start_pano = False

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret is True:
            frame = cv2.resize(frame, RESOLUTION)
            open_window("Live")
            cv2.imshow("Live", frame)

            if start_pano is True:
                frame_buffer.append(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("s"):
            if start_pano is True:
                print("Panorama being computed ...")
                video_panorama(frame_buffer,cam_matrix)
            start_pano = not start_pano

    cap.release()
    cv2.destroyAllWindows()

def live_motion_detection_demo(cap,cam_matrix):
    global FRAME_NB_BTW_PANO
    global PROJECTION_MATRICE

    ret = False
    frame = None

    start_live = False

    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    focal_length = cam_matrix[0][0]
    scaling_factor = focal_length #Scaling Factor equal to focal length
    PROJECTION_MATRICE = compute_projection_matrix(cam_matrix, scaling_factor, RESOLUTION)

    while(cap.isOpened()):
        prec_ret, prec_frame = (ret,frame)
        ret, frame = cap.read()

        if ret is True:
            frame = cv2.resize(frame, RESOLUTION)

        if prec_ret is True and ret is True and start_live:
            prec_gray = cv2.cvtColor(prec_frame,cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            motion_mask = motion_detection(fgbg,kernel, prec_gray, gray, PROJECTION_MATRICE, frame)

        if ret is True:
            open_window("Live")
            cv2.imshow("Live", frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("s"):
            start_live = not start_live

    cap.release()
    cv2.destroyAllWindows()

def live_enhanced_panorama(cap,cam_matrix):
    global PROJECTION_MATRICE
    global FRAME_NB_BTW_PANO

    frame_buffer = list()
    start_pano = False

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret is True:
            frame = cv2.resize(frame, RESOLUTION)
            open_window("Live")
            cv2.imshow("Live", frame)

            if start_pano is True:
                frame_buffer.append(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("s"):
            if start_pano is True:
                print("Panorama being computed ...")
                video_enhanced_panorama(frame_buffer,cam_matrix)
            start_pano = not start_pano

    cap.release()
    cv2.destroyAllWindows()

def live_personn_detection(cap):
    global MODEL_PATH

    if(PERSONN_DETECTION_ALGO == "Opencv"):
        detector = HumanDetectorOCV()
    elif(PERSONN_DETECTION_ALGO == "Tensorflow"):
        detector = HumanDetectorTF(MODEL_PATH)
    else:
        print("Error : Unknown Personn Detection algorithm")
        exit(-1)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret is True:
            frame = cv2.resize(frame, RESOLUTION)

            if start_detect is True:
                boxes = detector.detect(frame)
                for box in boxes:
                    cv2.rectangle(frame,box[0],box[1],(0,0,255),2)

            open_window("Live")
            cv2.imshow("Live", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("You quit.")
            break
        elif key == ord("s"):
            start_detect = not start_detect

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
        #cap = cv2.VideoCapture(0)
        cap = open_cam_onboard(WINDOW_WIDTH, WINDOW_HEIGHT, RESOLUTION,FRAME_RATE)

    elif(len(sys.argv) == 4):
        cmatrix_filename = sys.argv[1]
        cam_matrix = get_cam_matrix(cmatrix_filename)

        live = False
        mode = sys.argv[2]
        if mode not in IMPLEMENTED_MODE:
            print("Error: Implemented modes are " + str(IMPLEMENTED_MODE) + ".")
            exit(-1)
        video_dirname = sys.argv[3]
        nb_video = int(video_dirname.split("_")[1])
        perf_assess = False

        #cap = cv2.VideoCapture(video_filename)
        cap = frameReadingFromImage(video_dirname)

        if cap is None:
            print("Error: Fail to read the Video Files.")
            exit(-1)
    elif(len(sys.argv) == 5):
        cmatrix_filename = sys.argv[1]
        cam_matrix = get_cam_matrix(cmatrix_filename)

        live = False
        mode = sys.argv[2]
        if mode not in IMPLEMENTED_MODE:
            print("Error: Implemented modes are " + str(IMPLEMENTED_MODE) + ".")
            exit(-1)

        if mode == "motion_detection" or mode == "personn_detection":
            perf_assess = bool(sys.argv[4])
        else:
            print("Error: python3.6 main.py cam_matrix_filename.json mode=" + str(IMPLEMENTED_MODE) + " [video_dirname] [performance_assessment=True]")
            exit(-1)

        video_dirname = sys.argv[3]
        video_nb = int(video_dirname.split("_")[1])

        #cap = cv2.VideoCapture(video_filename)
        cap = frameReadingFromImage(video_dirname)

        if cap is None:
            print("Error: Fail to read the Video Files.")
            exit(-1)

    else:
        print("Error: python3.6 main.py cam_matrix_filename.json mode=" + str(IMPLEMENTED_MODE) + " [video_dirname] [performance_assessment=True]")
        exit(-1)

    if live is True and mode == "panorama":
        live_panorama(cap,cam_matrix)

    elif live is True and mode == "matching_demo":
        live_matching_demo(cap,cam_matrix)

    elif live is False and mode == "panorama":
        video_panorama(cap,cam_matrix)

    elif live is False and mode == "matching_demo":
        video_matching_demo(cap,cam_matrix)

    elif(live is False and mode == "motion_detection" and perf_assess == False):
        video_motion_detection_demo(cap,cam_matrix)

    elif(live is False and mode == "motion_detection" and perf_assess == True):
        motion_detection_assessment(cap,cam_matrix, video_nb)

    elif(live is True and mode == "motion_detection"):
        live_motion_detection_demo(cap,cam_matrix)

    elif(live is False and mode == "enhanced_panorama"):
        video_enhanced_panorama(cap,cam_matrix)

    elif(live is True and mode == "enhanced_panorama"):
        live_enhanced_panorama(cap,cam_matrix)

    elif(live is False and mode == "personn_detection" and perf_assess == False):
        video_personn_detection(video_dirname)

    elif(live is False and mode == "personn_detection" and perf_assess == False):
        personn_detection_assesment(video_dirname, video_nb)

    elif(live is True and mode == "personn_detection"):
        live_personn_detection(cap)
