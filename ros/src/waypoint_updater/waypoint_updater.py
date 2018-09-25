#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 9.81 * 1.5 # max deceleration 1.5 g

STOPLINE_OFFSET = 10    # make car stop before the stopline so that traffic light remains visible

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Only in simulator!!!!
        #rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_light_detector_cb)



        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.pose = None

        self.stopline_wp_idx = None
        self.stopline_dist = 100000

        self.loop()

    def loop(self):
        # 30 fps promises less lag for the waypoint generation
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # Get closest waypoint
                #closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coordinates
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx+1)%len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        lane = self.generate_lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        # replaced base_lane with base_waypoints
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        distance_to_stopline = 10000

        if self.stopline_wp_idx == None or (self.stopline_wp_idx >= farthest_idx) or self.stopline_dist >= 35:
            lane.waypoints = base_waypoints
        else:
            #rospy.loginfo("Will stop for traffic light!!!")
            #rospy.loginfo("stopline idx %d farthest_idx %d distance %.2f", self.stopline_wp_idx, farthest_idx, self.stopline_dist)
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        '''make the car slow down in front of a red traffic light or other obstacle'''
        temp = []
        # catch race condition in case light suddenly turns green and there is no
        # stopline anymore
        stopline = self.stopline_wp_idx
        if stopline == None:
            return waypoints
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            # two waypoints back from the line so that center of car doesn't go over line
            stop_idx = max(stopline - closest_idx - 1, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            #try shifted sigmoid function to slow down smoothly as a s-curve
            #vel = (math.tanh(dist / 8.0 - 2.) + 1.0)  * MAX_DECEL
            #rospy.loginfo("Setting new velocity %.2f", vel)
            if vel < 1.75:
                vel = 0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg


    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d =  [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_light_detector_cb(self, msg):
        '''callback for traffic light messages from TL detector '''
        tl_index = msg.data
        if tl_index != -1:
            # just set stoppline waypoint index
            self.stopline_wp_idx = tl_index - STOPLINE_OFFSET
            cx = self.pose.pose.position.x
            cy = self.pose.pose.position.y
            # light x/y
            lx,ly = self.waypoints_2d[tl_index]
            # euclidean distance
            dist = math.sqrt((cx-lx)**2 + (cy-ly)**2)

            rospy.loginfo("Setting next stopline_wp_idx at " + str(tl_index) + " at dist " + str(dist))
            self.stopline_dist = dist
        elif tl_index == -1:
            rospy.loginfo("Light not red, removing stopline")
            self.stopline_dist = 10000
            self.stopline_wp_idx = None

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # SIM ONLY !!!!!!
        light_info = msg.lights
        active_lights = []
        for light in light_info:
            pose = light.pose
            state = light.state
            #rospy.loginfo("Traffic Light at pos %.2f %.2f state: %d", pose.pose.position.x, pose.pose.position.y, state)
            # if yello or red --> mark for slowdown
            if state < 2:
                active_lights.append(light)


        if len(active_lights):
            #rospy.loginfo("Found %d red/yellow traffic lights", len(active_lights))
            # find closest active light, and save its index
            closest_light = []
            closest_dist = 100000

            for l in active_lights:
                # car x/y
                cx = self.pose.pose.position.x
                cy = self.pose.pose.position.y
                # light x/y
                lx = l.pose.pose.position.x
                ly = l.pose.pose.position.y
                # euclidean distance
                dist = math.sqrt((cx-lx)**2 + (cy-ly)**2)
                if dist < closest_dist:
                    closest_light = l
                    closest_dist = dist

            # now find the index of the traffic light on our path using the KDTree
            # and set it as self.stopline_wp_idx
            lx = closest_light.pose.pose.position.x
            ly = closest_light.pose.pose.position.y


            closest_light_idx = self.waypoint_tree.query([lx,ly],1)[1]
            #print cx,cy, closest_light_idx, lx, ly, closest_me_idx, len(self.waypoints_2d)
            #rospy.loginfo("Closest red or yellow traffic light at %.2f %.2f at %.2f m", lx, ly, closest_dist)
            #NOTE: closest_light - 5 in order to stop before light
            self.stopline_wp_idx = closest_light_idx - 5
            self.stopline_dist = closest_dist
        else:
            # no active traffic lights --> reset self.stopline_wp_idx to None
            self.stopline_wp_idx = None
            self.stopline_dist = 10000






    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
