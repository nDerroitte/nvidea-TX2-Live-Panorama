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

		(rect, weights) = self.hog.detectMultiScale(im, winStride=(4, 4), padding=(16, 16), scale=1.09)

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
			cv2.rectangle(im, (x + p_w, y + p_h), (x + w - p_w, y + h - p_h), (0, 255, 0), thickness)

			human_boxes.append([(int(boxes[i][1]), int(boxes[i][0])), (int(boxes[i][3]), int(boxes[i][2]))])

		# reformat and rearrange the boxes for convenience
		cv2.imshow("zeheff", im)
		cv2.waitKey(1)
		return human_boxes

if __name__ == "__main__":

	if len(sys.argv) == 2:
		video_path = sys.argv[1]
		files = sorted(os.listdir(video_path))
		detector = HumanDetectorOCV()
		start = time.time()

		for file_name in files:
			if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
				im = cv2.imread(video_path + file_name)
				im = imutils.resize(im, width=min(400, im.shape[1]))

				rect = detector.detect(im)

		end = time.time()
		print("Processing time :" + str(end - start))

	else:
		print("Must provide model and video path")
		exit()
