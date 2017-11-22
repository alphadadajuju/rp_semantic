#!/usr/bin/env python
# ROS imports
import roslib; roslib.load_manifest('rp_semantic')
import rospy
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import time
import itertools

import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
from rp_semantic.msg import Frame
from rp_semantic.msg import Cluster
from rp_semantic.msg import LabelClusters
from rp_semantic.msg import BoWP

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class ClusterVis:
    def __init__(self):
        print ('Program begins:')

    def start(self):

        self.rosSetup()
        while not rospy.is_shutdown():
            self.controlLoop()
        

    def rosSetup(self):

        # define class variables
        self.label_cluster_ready = False
        self.wait_for_message = True
        colours = '/home/alpha/catkin_ws/src/segnet_program/src/sun.png'
        colours_img = cv2.imread(colours)
        '''
        b,g,r = cv2.split(colours_img)       # get b,g,r
        colours_img = cv2.merge([r,g,b])     # switch it to rgb
        '''

        self.label_colours = colours_img.astype((np.uint8))
        #print('color test: ' + str(self.label_colours))

        print('color test: ' + str(self.label_colours[0][1][:]))

        self.labels_image = Image()
        self.bridge = CvBridge()

        # subscriber
        self.label_cluster_sub = rospy.Subscriber('/rp_semantic/labels_clusters', LabelClusters, self.label_cluster_callback, queue_size=1, buff_size=2**24) # change topic's name accordingly

        # publisher
        self.sphere_pub = rospy.Publisher('/rp_semantic/cluster_sphere', MarkerArray, queue_size=10) # change topic's name accordingly
        self.label2d_pub = rospy.Publisher('/rp_semantic/labels_image', Image, queue_size=10) # change topic's name accordingly
        self.pointcloud_pub = rospy.Publisher('/rp_semantic/pointcloud', PointCloud2, queue_size=10) # change topic's name accordingly
        self.rgb_pub = rospy.Publisher('/rp_semantic/rgb_image', Image, queue_size=10) # change topic's name accordingly

    def label_cluster_callback(self, label_cluster_msg):
        ''' 
        '''
        print ('message received')
        if self.wait_for_message == True:
            self.label_cluster = label_cluster_msg
            #self.node_id = self.label_cluster.node_id
            
            try:
                # bgr8 is the pixel encoding -- 8 bits per color, organized as blue/green/red
                self.label_2d = self.bridge.imgmsg_to_cv2(self.label_cluster.labels, "mono8")
                #self.rgb_2d = self.bridge.imgmsg_to_cv2(self.label_cluster.raw_rgb, "bgr8")
                #print(self.label_2d)
                #print(self.label_2d.shape)
                
            except CvBridgeError, e:
                # all print statements should use a rospy.log_ form, don't print!
                rospy.loginfo("Conversion failed")
            
            self.label_cluster_ready = True
            self.wait_for_message = False
            
   



    def controlLoop(self):
        #print ('enter control loop')
        #print (self.label_2d_ready)
        if self.label_cluster_ready is True:
            self.label_cluster_ready = False
            self.marker_sphere()
            rgb_im = self.label2rgb()
            self.labels_image = self.bridge.cv2_to_imgmsg(np.uint8(rgb_im), "bgr8")
            self.rgb_image = self.label_cluster.raw_rgb
            self.pointcloud = self.label_cluster.raw_pointcloud

            print('number of markers drawmn:' + str(np.shape(self.markerArray.markers)[0]))
            self.sphere_pub.publish(self.markerArray)
            self.label2d_pub.publish(self.labels_image)
            self.pointcloud_pub.publish(self.pointcloud)
            self.rgb_pub.publish(self.rgb_image)


            self.wait_for_message = True

            
  
            '''
            # To uncomment
            print(type(self.node_id))
            self.bowp_message.node_id = self.node_id
            self.bowp_message.bow = tuple(self.bow)
            self.bowp_message.bowp = tuple(self.bowp_flat)
            self.bowp_pub.publish(self.bowp_message)
            self.wait_for_message = True
            '''

    def marker_sphere(self):
        num_cluster = np.shape(self.label_cluster.clusters)[0]
        print (str(num_cluster) + ' clusters in this frame')
        
        m_delete  = Marker()
        m_delete.header.frame_id = "/camera_rgb_optical_frame"
        m_delete.action = m_delete.DELETEALL
        
        m_deletearray = MarkerArray()
        m_deletearray.markers.append(m_delete)
        self.sphere_pub.publish(m_deletearray)
        

        self.markerArray = MarkerArray()
        for i in range(0, num_cluster):
            marker = Marker()
            marker.header.frame_id = "/camera_rgb_optical_frame"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = self.label_cluster.clusters[i].radius
            marker.scale.y = self.label_cluster.clusters[i].radius
            marker.scale.z = self.label_cluster.clusters[i].radius
            
            marker.color.a = 0.6
            '''
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 0.0
            '''
            
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = self.label_cluster.clusters[i].x
            marker.pose.position.y = self.label_cluster.clusters[i].y
            marker.pose.position.z = self.label_cluster.clusters[i].z
            cluster_label = self.label_cluster.clusters[i].label
            marker.color.r = round(self.label_colours[0][cluster_label][2] / 255.0, 1)
            marker.color.g = round(self.label_colours[0][cluster_label][1] / 255.0, 1)
            marker.color.b = round(self.label_colours[0][cluster_label][0] / 255.0, 1)
            
            '''
            print('r:' + str(marker.color.r))
            print('g:' + str(marker.color.g))
            print('b:' + str(marker.color.b))
            '''
            
            self.markerArray.markers.append(marker)
            if i == num_cluster-1:
                id = 0
                for m in self.markerArray.markers:
                    m.id = id
                    id += 1



    def label2rgb(self):
        rgb_im = np.zeros((self.label_2d.shape[0], self.label_2d.shape[1], 3))
        #print(rgb_im.shape)
        for i in range(0,self.label_2d.shape[0]):
            for j in range(0, self.label_2d.shape[1]):
                pixel_label = self.label_2d[i][j]
                rgb_im[i][j][:] = self.label_colours[0][pixel_label]

        return rgb_im


            



            


if __name__ == '__main__':

    rospy.init_node( 'cluster_visualization', log_level=rospy.INFO)
    cluster_v_node = ClusterVis()
    cluster_v_node.start()