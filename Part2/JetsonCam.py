import cv2
 
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

def open_window(name, width = None, height = None):
    """
    Windows handling
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if width != None and height != None:
        cv2.resizeWindow(name, width, height)
