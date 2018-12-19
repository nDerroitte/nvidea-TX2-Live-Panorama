from HumanDetectorTF import HumanDetectorTF
from HumanDetectorOCV import HumanDetectorOCV
from BoxComp import *
import os, sys, time

VIDEO_PATH = "Capture/Video_1_2018-12-03_14_42_52/"
ANN_PATH = "Annotation/In/box_6_1.txt"

GRP_NB = 6
VIDEO_SEQ = 1
ANN_FOLDER = "Annotation/In/"

THRESHOLD = 0.8
MODEL_PATH = "models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"

THRESHOLDS = [0.6,0.7,0.8]
MODELS = [
    "models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb",
    "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    "models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
    "models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"]

def person_detection(algo, img, detector, p_a=False, img_nb=None
    , threshold = 0.8):
    """
    Perform the person detection using the appropriate method
    """

    start = time.time()
    if algo == 'tensorflow':
        boxes = detector.detect(img,threshold)
    if algo == 'opencv':
        boxes = detector.detect(img)
    detection_time = time.time() - start

    if p_a:
        boxes.sort(key=lambda x: x[0][0])
        error = boxComp(boxes, img_nb, GRP_NB,VIDEO_SEQ, ANN_FOLDER)

    for box in boxes:
        cv2.rectangle(img,box[0],box[1],(0,0,255),2)
    cv2.imshow("preview", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        exit()

    if p_a:
        return error, detection_time
    else:
        return detection_time

def detect_opcv(video_path):
    """
    Perform the person detection using the OpenCV method
    """
    files = sorted(os.listdir(video_path))
    detector = HumanDetectorOCV()

    for file_name in files:
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            image = cv2.imread(video_path + file_name)
            if image is not None:
                person_detection("opencv",image,detector)

def detect_tf(video_path,model_path,threshold=0.8):
    """
    Perform the person detection using the Tensorflow method
    """
    files = sorted(os.listdir(video_path))
    detector = HumanDetectorTF(model_path)

    for file_name in files:
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            image = cv2.imread(video_path + file_name)
            if image is not None:
                person_detection("tensorflow",image,detector,threshold=threshold)

def perf_ass_opcv(video_path,ann_path):
    """
    Perform the performance assessment of the OpenCV method
    """
    files = sorted(os.listdir(video_path))
    ref_img = getRefId(ann_path)
    detector = HumanDetectorOCV()

    img_nb=0
    ann_nb=0
    total_error = 0
    computation_time = 0

    for file_name in files:
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            image = cv2.imread(video_path + file_name)
            if image is not None:
                if img_nb in ref_img:
                    error, detection_time = person_detection("opencv",image,
                        detector,True,img_nb)
                    total_error += error
                    ann_nb += 1
                else:
                    detection_time = person_detection("opencv",image,detector)
                computation_time += detection_time
        img_nb += 1
    return (total_error/(ann_nb+1)), (computation_time/(img_nb+1))

def perf_ass_tf(video_path, ann_path, model_path, threshold):
    """
    Perform the performance assessment of the Tensorflow method
    """
    files = sorted(os.listdir(video_path))
    ref_img = getRefId(ann_path)
    detector = HumanDetectorTF(model_path)

    img_nb=0
    ann_nb=0
    total_error = 0
    computation_time = 0

    for file_name in files:
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            image = cv2.imread(video_path + file_name)
            if image is not None:
                if img_nb in ref_img:
                    error, detection_time = person_detection("opencv",image,
                        detector,True,img_nb,threshold)
                    total_error += error
                    ann_nb += 1
                else:
                    detection_time = person_detection("opencv",image,detector
                        ,threshold=threshold)
                computation_time += detection_time
        img_nb += 1
    return (total_error/(ann_nb+1)), (computation_time/(img_nb+1))

def study_params(video_path, ann_path, models, thresholds):
    """
    Study the parameters of the Tensorflow method
    """
    for model_path in models:
        for threshold in thresholds:
            error, computation_time = perf_ass_tf(video_path, ann_path
                , model_path, threshold)
            print("\nmodel : " + model_path)
            print("threshold : " + str(threshold))
            print("error : " + str(error))
            print("computation time : " + str(computation_time))
