import cv2
import datetime
import os
import sys

###############################################################################
#                                  Constants                                  #
###############################################################################
WINDOW_NAME = 'CameraDemo'
CAPTUREPATH = os.path.abspath("Capture/") + "/"
RESOLUTION = (1280,720)
WINDOW_WIDTH=1280
WINDOW_HEIGHT=720
FRAME_RATE = 25
GROUP_NUMBER = 6

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

def record_frame(video_sequence_nb, frame_buffer, ref):
    """
    Creating the images from the frames recorded.
    The writing of image is make outside the main loop to safe time
    """
    # Creating the complete name of the image
    current_video_path = (CAPTUREPATH + "Video_" + str(video_sequence_nb) + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/")

    # Create the folder Capture if it does not exist
    if not os.path.isdir(CAPTUREPATH):
        create_directory(CAPTUREPATH)
    # Create the folder for the current name of the sequence
    if not os.path.isdir(current_video_path):
        create_directory(current_video_path)

    # Writing the images
    index = 0
    for frame in frame_buffer:
        if ref is True:
                result = cv2.imwrite((current_video_path + "/ref_" + str(GROUP_NUMBER) + "_" + str(video_sequence_nb) + "_" + str(index) + ".jpg"),frame)
        else:
                result = cv2.imwrite((current_video_path + "/img_" + str(GROUP_NUMBER) + "_" + str(video_sequence_nb) + "_" + str(index) + ".jpg"),frame)
        if ret is True:
            if result is False:
                print("Fail to Save the Image.\n")
        else:
            print("Fail to Save the Image.\n")
        index = index + 1

###############################################################################
#                                  Main                                       #
###############################################################################
if __name__ == "__main__":
    """
    Checking the arguements number :
        id for id of the sequence (1 indoor, 2 outdoor)
        [True,False] = reference sequence or not
    """

    if(len(sys.argv) < 3):
        print("Error : You must specify the Video Sequence Id and ref = [True,False].")
        exit(-1)
    video_sequence_nb = sys.argv[1]
    ref = bool(sys.argv[2])

    # Openning camera
    cap = open_cam_onboard(WINDOW_WIDTH,WINDOW_HEIGHT,RESOLUTION,FRAME_RATE)
    # Windows oppening
    open_window(WINDOW_WIDTH,WINDOW_HEIGHT)

    # Variables initialisation
    record = False
    frame_buffer = list()
    current_video_path = None
    frame_to_record = None
    stop_recording = False

    while(cap.isOpened()):
        # Capture frames
        ret, frame = cap.read()
        # Operations on frames
        if ret is True:
            if record is True:
                # We are currently recording
                # Computing the gray image for display purpose
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Add the frame (color) in the buffer to write
                frame_buffer.append(frame)
                # frame count
                captured_frame = captured_frame + 1
                # Ending condition and final printing
                if frame_to_record == captured_frame:
                    stop_recording = datetime.datetime.now()
                    record_frame(video_sequence_nb, frame_buffer,ref)
                    print("Stop Recording.")
                    print("You captured " + str(captured_frame) + " frame.")
                    print("You record during " + str(stop_recording - start_recording) + ".")
                    frame_to_record = None
                    record = False

                # While recording, display gray image and show time of recording
                cv2.putText(gray,str(datetime.datetime.now() - start_recording), (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255))
                cv2.imshow(WINDOW_NAME,gray)

            else:
                # If we are not recording, just display the gray image
                cv2.imshow(WINDOW_NAME,frame)

        key = cv2.waitKey(1) & 0xFF
        # Main user input reading
        if key == ord("p"):
            # Capturing a single image
            if not os.path.isdir(CAPTUREPATH):
                create_directory(CAPTUREPATH)
            # Writing the image
            path = CAPTUREPATH + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".png"
            result = cv2.imwrite(path,frame)
            if ret is True:
                if result is False:
                    print("Fail to Save the Image.\n")
                else:
                    print("Image saved in CalibrationImage Folder.\n")
            else:
                print("Fail to Save the Image.\n")
        elif key == ord("v"):
            # 1500 frames case
            # If we repress "v" while it's already recording,
            # we stop the recording manualy
            if record is True:
                # Stopping the record
                stop_recording = datetime.datetime.now()
                record_frame(video_sequence_nb, frame_buffer, ref)
                print("Stop Recording.")
                print("You captured " + str(captured_frame) + " frames.")
                print("You record during " + str(stop_recording - start_recording) + ".")
                frame_to_record = None
            else:
                # Recording 1500 frames
                print("Start Recording ...")
                frame_to_record = 1500
                start_recording = datetime.datetime.now()
                captured_frame = 0
            record = not record
        elif key == ord("d"):
            # 500 frames case
            # If we repress "v" while it's already recording,
            # we stop the recording manualy
            if record is True:
                # Stopping the record
                stop_recording = datetime.datetime.now()
                record_frame(video_sequence_nb,frame_buffer, ref)
                print("Stop Recording.")
                print("You captured " + str(captured_frame) + " frames.")
                print("You record during " + str(stop_recording - start_recording) + ".")
                frame_to_record = None
            else:
                # Recording 500 frames
                print("Start Recording ...")
                frame_to_record = 500
                start_recording = datetime.datetime.now()
                captured_frame = 0

            record = not record
        elif key == ord("q"):
            # Quit
            break

    # Closing camera and window
    cap.release()
    cv2.destroyAllWindows()
