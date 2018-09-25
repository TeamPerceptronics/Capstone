#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        try:
            self.save_false = self.config['save_false_detections']
        except KeyError:
            self.save_false = False

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.STATE_MAP = {
            TrafficLight.UNKNOWN: 'unknown',
            TrafficLight.GREEN: 'green',
            TrafficLight.RED: 'red',
            TrafficLight.YELLOW: 'yellow'
        }

        self.f_cnt = 0
        rospy.loginfo("TL Detector Initialized!!!")
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        #rospy.loginfo("Image_CB")

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''


        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == 0 else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def distance(self, pose1, pose2):
        """ Gets the distance between two pose vectors
        """
        dx = pose1.x - pose2.x
        dy = pose1.y - pose2.y
        dz = pose1.z - pose2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO A simple implementation. May be upgraded
        #NOTE better to use KD Tree
        index = 0
        mindist = 99999
        for i, wp in enumerate(self.waypoints.waypoints):
            # rospy.loginfo('pose: {0}, wp.pose.pose: {1}'.format(pose.position, wp.pose.pose.position))
            dist = self.distance(pose.position, wp.pose.pose.position)
            if dist < mindist:
                mindist = dist
                index = i
        return index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        state = self.light_classifier.get_classification(cv_image)
        if self.save_false and state != light.state:
            cv2.imwrite('incorrect/sim_img_{0}.png'.format(self.f_cnt), cv_image)
            self.f_cnt += 1
        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        max_dist_to_light = 50      # meters

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            # Get the waypoint closest to our position
            car_position = self.get_closest_waypoint(self.pose.pose)
            # rospy.loginfo('Closest car waypoint: {0}'.format(car_position))

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                line_pose = Pose()
                line_pose.position.x = line[0]
                line_pose.position.y = line[1]
                temp_wp_idx = self.get_closest_waypoint(line_pose)
                # Find closest stop ine waypoint index
                d = temp_wp_idx - car_position
                dist_to_light = self.distance(self.pose.pose.position, line_pose.position)
                #rospy.loginfo('Distance to light: {0}'.format(dist_to_light))
                if d >= 0 and d < diff and dist_to_light < max_dist_to_light:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        rospy.loginfo('Closest light waypoint {0}'.format(line_wp_idx))
        state = TrafficLight.UNKNOWN
        if closest_light:
            state = self.get_light_state(closest_light)

            #rospy.loginfo('Expected light state is {0}: {1}'.format(closest_light.state, self.STATE_MAP[closest_light.state]))
            rospy.loginfo('raw state: ' + str(state))
            #return -1, TrafficLight.UNKNOWN #light_wp, state

        #NOTE: Sven removed this as message would result in always -1
        #light_wp = -1
        # if line_wp_idx is not None:
        #     if closest_light.state == TrafficLight.RED:
        #         rospy.loginfo("Setting traffic light RED light_wp " + str(state) )
        #         light_wp = line_wp_idx
        #     else:
        #         light_wp = -1

        #self.waypoints = None
        return line_wp_idx, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
