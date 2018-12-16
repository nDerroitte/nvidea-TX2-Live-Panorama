import re
import glob
import cv2
import numpy as np
from os import rename


###############################################################################
#                                  Fonctions                                  #
###############################################################################

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

def renameImg(folder_name):
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
    for i in range (len(images_names)):
        digit = str(i).zfill(4)
        final_name = "img_6_1_" + digit +".jpg"
        #final_name = "img_6_2_" + digit +".jpg"
        #final_name = "ref_6_1_" + digit +".jpg"
        #final_name = "ref_6_2_" + digit +".jpg"

        print("Renaming image: {} to {}".format(digit,final_name))
        rename(images_names[i],final_name)


if __name__ == "__main__":
    renameImg("./")
