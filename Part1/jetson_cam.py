import cv2

def open_cam_onboard(width, height):
    pass

def open_window(name, width = None, height = None):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if width != None and height != None:
        cv2.resizeWindow(name, width, height)
