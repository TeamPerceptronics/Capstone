import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.graph = tf.Graph()
        FROZEN_GRAPH = 'models/frozen_inference_graph.pb'

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_GRAPH, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            
            self.sess = tf.Session(graph=self.graph)

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            # Output tensor names compatible with TF Object Detection API
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        
        rospy.loginfo('Classifier model loaded')
        
        self.STATE_MAP = {
            0: 'unknown',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'off'
        }
            
        # pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # Add batch axis to the image
        image = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            (boxes, scores, classes) = self.sess.run([self.boxes, self.scores, self.classes],
                feed_dict = {self.image_tensor: image}
            )
        
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        light_state = TrafficLight.UNKNOWN
        if scores[0] > 0.9 and classes[0] == 1:
            light_state = TrafficLight.GREEN
        elif scores[0] and classes[0] == 2:
            light_state = TrafficLight.RED
        elif scores[0] and classes[0] == 3:
            light_state = TrafficLight.YELLOW
        
        rospy.loginfo('Detected light state: {0} with score {1}'.format(self.STATE_MAP[classes[0]], scores[0]))
        return light_state
