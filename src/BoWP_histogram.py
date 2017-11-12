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

class BoWP_hist:
    def __init__(self):
        print ('Program begins:')

    def start(self):

        self.rosSetup()
        while not rospy.is_shutdown():
            self.controlLoop()
        

    def rosSetup(self):

        # define class variables
        self.bridge = CvBridge()
        
        self.num_class = rospy.get_param("rp_semantic/BOWP_histogram/num_class",37) 
        tup = (1, )
        for i in range(2, self.num_class+1): # number of classes
            tup2 = (i, )
            tup = tup + tup2
        
        self.pairs = list(itertools.permutations(tup, 2))
        #self.bowp = np.zeros((np.shape(self.pairs)[0],), dtype=np.float64) # empty bowp array
        self.bowp = np.zeros((self.num_class, self.num_class), dtype=np.float64)
        print('Number of word-pairs: ' + str(self.bowp.shape[0]))

        self.wait_for_message = rospy.get_param("rp_semantic/BOWP_histogram/wait_for_message",True) 
        self.label_2d_ready = rospy.get_param("rp_semantic/BOWP_histogram/label_2d_ready",False)
        self.label_2d = rospy.get_param("rp_semantic/BOWP_histogram/label_2d",None)
        self.label_cluster_msg = rospy.get_param("rp_semantic/BOWP_histogram/label_cluster_ms",None)
        #print(self.pairs)
        #indices =  self.pairs.index((4,3))
        #print(indices)

        # To uncomment
        self.bowp_message = BoWP()

        
        # subscriber
        self.label_cluster_sub = rospy.Subscriber('/rp_semantic/labels_clusters', LabelClusters, self.label_cluster_callback, queue_size=1, buff_size=2**24) # change topic's name accordingly

        # To uncomment
        # publisher
        self.bowp_pub = rospy.Publisher('/rp_semantic/bow_bowp_descriptors', BoWP, queue_size=1) # change topic's name accordingly


    def label_cluster_callback(self, label_cluster_msg):
        ''' 
        '''
        print ('message received')
        if self.wait_for_message == True:
            self.label_cluster_msg = label_cluster_msg
            self.node_id = self.label_cluster_msg.node_id
            
            try:
                # bgr8 is the pixel encoding -- 8 bits per color, organized as blue/green/red
                self.label_2d = self.bridge.imgmsg_to_cv2(self.label_cluster_msg.labels, "mono8")
                self.label_2d_ready = True
                self.wait_for_message = False
                
            except CvBridgeError, e:
                # all print statements should use a rospy.log_ form, don't print!
                rospy.loginfo("Conversion failed")

        '''
        try:
            # bgr8 is the pixel encoding -- 8 bits per color, organized as blue/green/red
            self.label_2d = self.bridge.imgmsg_to_cv2(label_cluster_msg.label, "mono8")
            self.label_2d_ready = True
            #print(self.label_2d)
        except CvBridgeError, e:
            # all print statements should use a rospy.log_ form, don't print!
            rospy.loginfo("Conversion failed")

        
        if np.any(self.label_2d > 37):
            print('bad news')
        else:
            print("good news")
        
        # TO uncomment; no data to verify yet!
        #self.node_id = label_cluster_msg.node_id
        #self.label_cluster = label_cluster_msg.node_id
        '''

    def controlLoop(self):
        #print ('enter control loop')
        #print (self.label_2d_ready)
        if self.label_2d_ready is True:
            self.label_2d_ready = False
            # lock to local variables in case some class variables get updated faster than others
            #this_label_cluster = self.self.label_cluster
            self.bow = self.bow_hist()
            self.bowp_flat = self.bowp_hist()
            #print(self.bowp_flat.shape) #[]float64
            #print(self.bowp_flat) #[]float64
            #this_bowp_hist = plt.hist(self.bowp_flat, bins= self.num_class , range=(1, self.num_class**2))
            #plt.show(block=True)
            #input("Press Enter to continue...")

            
            self.bowp = np.zeros((self.num_class, self.num_class), dtype=np.float64)

            # To uncomment
            print(type(self.node_id))
            self.bowp_message.node_id = self.node_id
            self.bowp_message.bow = tuple(self.bow)
            self.bowp_message.bowp = tuple(self.bowp_flat)

            # Adding extra info for RANSAC and other
            self.bowp_message.clusters = self.label_cluster_msg.clusters
            self.bowp_message.raw_rgb = self.label_cluster_msg.raw_rgb
            self.bowp_message.raw_pointcloud = self.label_cluster_msg.raw_pointcloud

            self.bowp_pub.publish(self.bowp_message)
            self.wait_for_message = True




    def bow_hist(self):
        label_1d = np.squeeze(np.resize(self.label_2d, (1, 172800))) # width x height
        this_bow_hist = plt.hist(label_1d, bins= self.num_class , range=(1, self.num_class))
        this_bow_hist = this_bow_hist[0]

        this_bow_hist[1] = this_bow_hist[2] = this_bow_hist[22] = 0
        return (this_bow_hist)

    def bowp_hist(self):
        count = 0
        num_cluster = np.shape(self.label_cluster_msg.clusters)[0] # num of clusters/objects in a node
        print (str(num_cluster) + ' clusters in this frame')
        for i in range(0, num_cluster):
            for j in range(i, num_cluster):
                if i != j:
                    if self.label_cluster_msg.clusters[i].label == 1 or \
                            self.label_cluster_msg.clusters[i].label == 2 or \
                            self.label_cluster_msg.clusters[i].label == 22:
                        continue

                    #print ('i:' + str(i))
                    #print ('j:' + str(j))
                    p2p_dist = ((self.label_cluster_msg.clusters[i].x - self.label_cluster_msg.clusters[j].x) ** 2 + (self.label_cluster_msg.clusters[i].y - self.label_cluster_msg.clusters[j].y) ** 2 + (self.label_cluster_msg.clusters[i].z - self.label_cluster_msg.clusters[j].z) ** 2) ** 0.5

                    if self.label_cluster_msg.clusters[i].radius > p2p_dist:
                        #indice =  self.pairs.index((self.label_cluster.clusters[i].label,self.label_cluster.clusters[j].label))
                        count += 1
                        #print('count = ' + str(count))
                        #print('i label: ' + str(np.uint8(self.label_cluster.clusters[i].label)))
                        #print('j label: ' + str(np.uint8(self.label_cluster.clusters[j].label)))
                        self.bowp[np.uint8(self.label_cluster_msg.clusters[i].label)][np.uint8(self.label_cluster_msg.clusters[j].label)] += 1
                        print (self.bowp[np.uint8(self.label_cluster_msg.clusters[i].label)][np.uint8(self.label_cluster_msg.clusters[j].label)])

                    if self.label_cluster_msg.clusters[j].radius > p2p_dist:
                        #indice =  self.pairs.index((self.label_cluster.clusters[j].label,self.label_cluster.clusters[i].label))
                        #print('i2 label: ' + str(np.uint8(self.label_cluster.clusters[i].label)))
                        #print('j2 label: ' + str(np.uint8(self.label_cluster.clusters[j].label)))
                        self.bowp[np.uint8(self.label_cluster_msg.clusters[j].label)][np.uint8(self.label_cluster_msg.clusters[i].label)] += 1
                    #print('combination:' + str(i) + ', ' + str(j))
                    
        #print (self.bowp)
        return np.reshape(self.bowp, (np.product(self.bowp.shape),))
                
            


if __name__ == '__main__':

    rospy.init_node( 'bowp_hist', log_level=rospy.INFO)
    bowp_hist_node = BoWP_hist()
    bowp_hist_node.start()
