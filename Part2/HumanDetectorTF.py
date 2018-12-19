import numpy as np
import tensorflow as tf
import cv2

HUMAN_CLASS = 1

class HumanDetectorTF:
    def __init__(self, model_path):
        """
        Initialize the human detector object

        Parameters
        ----------
        model_path: string
            The path to the model file
        """

        # initilize the variables of the detector
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            # load the model from file
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # initialize the tensorflow session and the image tensor
        self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # create the tensor dictionary
        self.tensor_dict = {}
        ops = detection_graph.get_operations()
        tensor_names = {output.name for op in ops for output in op.outputs}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in tensor_names:
                self.tensor_dict[key] = detection_graph.get_tensor_by_name(
                    tensor_name)

    def detect(self,image,threshold=0.8):
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

        # expand the image and run the detection on it
        output = self.sess.run(self.tensor_dict,
            feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # retrieve the predicted class and thier associated score of confidence
        # as well as thier corresponding boxe locations
        detection_classes = output['detection_classes'][0].astype(np.uint8)
        detection_scores = output['detection_scores'][0]
        detection_boxes = output['detection_boxes'][0]

        human_boxes = []
        img_height, img_width,_ = image.shape
        for i in range(len(detection_boxes)):
            # keep only the boxes of the human class
            if (detection_classes[i] == HUMAN_CLASS
                # check the confidence score
                and detection_scores[i] > threshold):

                # reformat and rearrange the boxes for convenience
                human_boxes.append([(int(detection_boxes[i][1]*img_width),
                int(detection_boxes[i][0]*img_height)),
                (int(detection_boxes[i][3]*img_width),
                int(detection_boxes[i][2]*img_height))])

        return human_boxes
