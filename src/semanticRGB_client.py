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


from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from rp_semantic.srv import *
class SegnetSemantic:
    def __init__(self):
        print ('Semantic initialized:')

    def start(self):

        self.rosSetup()
        while not rospy.is_shutdown():
            self.controlLoop()
        

    def rosSetup(self):
        self.raw_image = None
        self.image_ready = False
        self.prob_message = Float64MultiArray()
        self.bridge = CvBridge() # for decoding sensor_msgs Image data[]
        #self.s = rospy.Service('rgb_to_label_prob', RGB2LabelProb, self.handle_rgb_to_label_prob)
        
        # subscriber
        self.rgb_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback, queue_size=1, buff_size=2**24) # change topic's name accordingly

        # publisher
        self.semantic_prob_pub = rospy.Publisher('/rp_semantic/labels_prob', Float64MultiArray, queue_size=1) # change topic's name accordingly
        
    def rgb_callback(self, rgb_msg):
        if self.image_ready == False:
            self.raw_image = rgb_msg
            self.image_ready = True

    def controlLoop(self):
        if self.image_ready == True:
            
            print 'wait for response from server'
            rospy.wait_for_service('rgb_to_label_prob')
            try:
                rgb_to_label_prob = rospy.ServiceProxy('rgb_to_label_prob', RGB2LabelProb)
                response = rgb_to_label_prob(self.raw_image)
                self.prob_message = response.image_class_probability
                #print 'response_message:', self.prob_message
                self.semantic_prob_pub.publish(self.prob_message)
                self.image_ready = False
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e
                

if __name__ == '__main__':

    rospy.init_node( 'semanticRGB_client', log_level=rospy.INFO)
    seg_node = SegnetSemantic()
    seg_node.start()
