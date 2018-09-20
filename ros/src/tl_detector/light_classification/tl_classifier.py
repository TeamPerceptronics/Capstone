import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.graph = tf.Graph()
        FROZEN_GRAPH = '/home/md/Projects/Udacity/Capstone_downloads/models/train_ssd_sim/fine_tuned/frozen_inference_graph.pb'

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
        if classes[0] == 1:
            light_state = TrafficLight.GREEN
        elif classes[0] == 2:
            light_state = TrafficLight.RED
        elif classes[0] == 3:
            light_state = TrafficLight.YELLOW
        
        rospy.loginfo('Detected light state: {0}'.format(light_state))
        return light_state
