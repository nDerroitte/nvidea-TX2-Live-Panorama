import cv2
from JetsonCam import *

###############################################################################
#                                  Constants                                  #
###############################################################################
#MATCHING_METHOD = "bruteforce"
MATCHING_METHOD = "flann"

FEATURE_DETECTOR = "orb"
#FEATURE_DETECTOR = "sift"
#FEATURE_DETECTOR = "surf"

###############################################################################
#                                  Fonctions                                  #
###############################################################################
def match_images(img1, img2, display):
    """
    Find the matches between 2 images
    """
    global MATCHING_METHOD

    if MATCHING_METHOD == 'bruteforce':
        return bruteforce_matching(img1, img2, display)
    elif MATCHING_METHOD == 'flann':
        return flann_matching(img1, img2, display)
    else:
        print("Error : Unknown Feature Matching Method : " + method + ".")
        exit(-1)

def bruteforce_matching(img1, img2, display, method = 'default'):
    """
    Brute force matching :
    It will find the best matches as it will try combinaison of all

    """
    global FEATURE_DETECTOR
    feature_detector = get_feature_detector(FEATURE_DETECTOR)

    #We find the keypoints(features) and descriptors
    key_points1, descriptors1 = feature_detector.detectAndCompute(img1, None)
    key_points2, descriptors2 = feature_detector.detectAndCompute(img2, None)

    # Match descriptors.
    if method == 'default':
        # create BFMatcher object with distance measurements NORM_HAMMING --> Brute Force
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        img3 = None
        matches = bf.match(descriptors1,descriptors2)
        # Sort them in the order of their distance (best matches with low distances are first)
        matches = sorted(matches, key = lambda x:x.distance)
        good_to_display = matches

        if display is True:
            img3 = cv2.drawMatches(img1,key_points1,img2,key_points2,good_to_display,img3, flags=2)

    elif method == 'knn':
        # create BFMatcher object with distance measurements NORM_HAMMING --> Brute Force
        # crosscheck is an alternative to the Ratio Test --> can't use knnMatch with crosscheck
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        img3 = None
        matches = bf.knnMatch(descriptors1,descriptors2, k = 2)
        # ratio test --> See D.Lowe's paper
        good_to_display = []
        good_to_return = []

        #Ratio Test
        for i in matches:
            if(len(i) == 2):
                m = i[0]
                n = i[1]
                if m.distance < 0.75*n.distance:
                    good_to_display.append([m])
                    good_to_return.append(m)

        matches = good_to_return

        if display is True:
            img3 = cv2.drawMatchesKnn(img1,key_points1,img2,key_points2,good_to_display,img3, flags=2)
    else:
        print("Error : Unknown Bruteforce Matching Method : " + method + ".")
        exit(-1)

    if display is True:
        open_window("Feature Matcher - " + method + " Bruteforce")
        cv2.imshow("Feature Matcher - " + method + " Bruteforce",img3)

    return matches, key_points1, key_points2

def flann_matching(img1, img2, display, method = "orb"):
    """
    Flann matching : ill find an approximate nearest neighbor
    It will find a good matching, but not necessarily the best possible one.
    """
    global FEATURE_DETECTOR
    feature_detector = get_feature_detector(FEATURE_DETECTOR)

    #We find the keypoints(features) and descriptors
    key_points1, descriptors1 = feature_detector.detectAndCompute(img1, None)
    key_points2, descriptors2 = feature_detector.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6

    if method == "orb":
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6,
                key_size = 12,
                multi_probe_level = 1)
        #Specify the number of times the trees should be recursively traversed
    elif method == "sift" or method == "surf":
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # or pass empty dictionary

    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)

    # ratio test --> See D.Lowe's paper
    good_to_display = []
    good_to_return = []

    for i in matches:
        if(len(i) == 2):
            m = i[0]
            n = i[1]
            if m.distance < 0.75*n.distance:
                good_to_display.append([m])
                good_to_return.append(m)

    matches = good_to_return
    if display is True:
        img3 = None
        img3 = cv2.drawMatchesKnn(img1,key_points1,img2,key_points2,good_to_display,img3, flags=2)

        open_window("Feature Matcher - " + method + " Flanner")
        cv2.imshow("Feature Matcher - " + method + " Flanner", img3)

    return matches, key_points1, key_points2

def get_feature_detector(method):
    """
    Get the feature detector of the method chosen
    """
    if method == "orb":
        return cv2.ORB_create()
    elif method == "sift":
        return cv2.xfeatures2d.SIFT_create()
    elif method == "surf":
        return cv2.xfeatures2d.SURF_create()
    else:
        print("Error: Unknown Feature Detector Method : " + method + ".")
        exit(-1)
