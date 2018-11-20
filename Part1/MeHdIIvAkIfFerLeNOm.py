import re
import glob
import cv2
import numpy as np



def atoi(text):
    """
    Fonction used to make the natural sort of the files name in the foler.
    Source : https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    Author :unutbu
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    Fonction used to make the natural sort of the files name in the foler.
    Source : https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    Author :unutbu
    """
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def frameReadingFromImage(folder_name):
    """
    folder_name = name of the directory containing the files.
    Return : list of frame, one for each image file.
    """
    list_frame = []
    #Reading the files names
    images_names = glob.glob(folder_name+"*.jpg")
    #Natural sort of the files names
    images_names.sort(key = natural_keys)
    #Writing frame, one for each jgp image.
    for image in images_names:
        list_frame.append(cv2.imread(image,0))

    return list_frame

if __name__ == "__main__":
    folder_name = "Capture/"
    list = frameReadingFromImage(folder_name)
    #print(list)
