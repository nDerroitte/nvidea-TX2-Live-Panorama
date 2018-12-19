from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils, cv2, os, argparse, sys, time


class HumanDetectorOCV:
	def __init__(self):
		"""
		Initialize the human detector object
		Parameters
		----------
		model_path: stringself.detect(im)
			The path to the model file
		"""
		self.hog = cv2.HOGDescriptor()
		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	def detect(self, image):
		"""
		Detect the humans in the image
		Parameters
		----------
		image: numpy 2D array
			The image on which perform the detection
		threshold : float
			The threshold for the detection confidence

		Return
		------
		human_boxes: list of array of 2 points
			The list of boxes constructed around the detected humans
		"""

		(rect, weights) = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(24, 24), scale=1.05)

		# non-maxima suppression applied to the boxes
		rect = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rect])
		boxes = non_max_suppression(rect, probs=None, overlapThresh=0.3)

		# draw of the rectangles around people
		human_boxes = []
		thickness = 1
		for i in range(len(boxes)):
			x = boxes[i][0]
			y = boxes[i][1]
			w = boxes[i][2]
			h = boxes[i][3]
			p_w, p_h = int(0.1*w), int(0.025*h)

			human_boxes.append([(x + p_w, y + p_h), (x + w - p_w, y + h - p_h)])

		# reformat and rearrange the boxes for convenience
		human_boxes.sort(key=lambda x: x[0][0])
		return human_boxes
