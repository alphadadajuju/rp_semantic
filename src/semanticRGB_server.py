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

import caffe


import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
from rp_semantic.msg import Frame
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension

from rp_semantic.srv import *
class SegnetSemantic:
    def __init__(self):
        print ('Semantic initialized:')
        

    def start(self):

        self.rosSetup()

        while not rospy.is_shutdown():
            self.controlLoop()
        

    def rosSetup(self):

        #caffe_root = '/home/alpha/github/caffe-segnet-cudnn5/'
        caffe_root = '/home/albert/GitHub/caffe-segnet-cudnn5/'

        #model_path = '/home/alpha/catkin_ws/src/segnet_program/src/'
        model_path = '/home/albert/rp_data/'

        # initialize segnet 
        sys.path.append('/usr/local/lib/python2.7/site-packages')
        sys.path.insert(0, caffe_root + 'python')
        caffe.set_mode_gpu()
        caffe.set_device(0) 

        low_res = False

        if low_res:
            model = model_path + 'segnet_sun_low_resolution.prototxt' # runtime error if do not include full path
            weights = model_path + 'segnet_sun_low_resolution.caffemodel'
        else:
            model = model_path + 'segnet_sun.prototxt'
            weights = model_path + 'segnet_sun.caffemodel'

        colours = model_path + 'sun.png'
        self.net = caffe.Net(model, weights, caffe.TEST)
        
        self.input_shape = self.net.blobs['data'].data.shape # 1 x 3 x 224 z 224
        self.output_shape = self.net.blobs['argmax'].data.shape # 1 x 1 x 224 z 224
        
        # no need for color info; will remove this soon
        '''
        colours_img = cv2.imread(colours)


        if colours_img is None:
            exit()
  
        self.label_colours = colours_img.astype(np.uint8)
        '''

        # class variable initialization
        self.rgb_frame = None
        self.height = 480
        self.width = 640
        self.num_class = 37

        self.bridge = CvBridge() # for decoding sensor_msgs Image data[]

        # needed initialization for multiarray (3 dimension) message
        self.class_response = Float64MultiArray()
        self.class_response.data = [0]* self.num_class * self.width * self.height 
        self.class_response.layout.dim.append(MultiArrayDimension()) # append 3 times to initialize a 3 dimensional multiarray
        self.class_response.layout.dim.append(MultiArrayDimension())
        self.class_response.layout.dim.append(MultiArrayDimension())

        # initialize based on multiarrayLayout definition (see online)
        self.class_response.layout.data_offset = 0
        self.class_response.layout.dim[0].label = "height"
        self.class_response.layout.dim[0].size = self.height
        self.class_response.layout.dim[0].stride = self.height * self.width * self.num_class
        self.class_response.layout.dim[1].label = "width"
        self.class_response.layout.dim[1].size = self.width
        self.class_response.layout.dim[1].stride = self.width * self.num_class
        self.class_response.layout.dim[2].label = "class"
        self.class_response.layout.dim[2].size = self.num_class
        self.class_response.layout.dim[2].stride = self.num_class

        self.dstride1 = self.class_response.layout.dim[1].stride 
        self.dstride2 = self.class_response.layout.dim[2].stride 

        # ROS service server
        self.s = rospy.Service('rgb_to_label_prob', RGB2LabelProb, self.handle_rgb_to_label_prob)

        # flags to exchange processed data between service handler and main control loop (faster!)
        self.wait_for_segnet = False
        self.obtain_rgb = False


    def handle_rgb_to_label_prob(self, req):

        print 'receive a request!'
        
        if self.wait_for_segnet is False: # if main control loop not busy
            try:
            # bgr8 is the pixel encoding -- 8 bits per color, organized as blue/green/red
                self.rgb_frame = self.bridge.imgmsg_to_cv2(req.rgb_image, "bgr8")
                self.obtain_rgb = True
                self.wait_for_segnet = True # main control loop executes segnet and ...
                print 'obtain rgb frame'
            except CvBridgeError, e:
            # all print statements should use a rospy.log_ form, don't print!
                    rospy.loginfo("Conversion failed")

            while self.wait_for_segnet is True: # main control loop still processing segnet and wrapping multiarray
                pass

            return RGB2LabelProbResponse(self.class_response)
        '''
        self.rgb_frame = cv2.resize(self.rgb_frame, (self.input_shape[3], self.input_shape[2]))

        input_image = self.rgb_frame.transpose((2, 0, 1))
        input_image = np.asarray([input_image])
        #print('shape' + str(input_image.shape))
        # run through Segnet
        while input_image.shape[3] != self.input_shape[3]:
            input_image = self.rgb_frame.transpose((2, 0, 1))

        print 'begin segnet'
        start = time.time()
        out = self.net.forward_all(data=input_image)
        end = time.time()
        print 'finish segnet'
        print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'
        print 'filling in class prob!'
        start = time.time()
        for row in range(0, self.height):
            for col in range(0, self.width):
                for cl in range (0,self.num_class):
                    self.class_response.data[self.dstride1*row + self.dstride2*col + cl] = self.net.blobs['conv1_1_D'].data[:,cl+1,row,col]
        end = time.time()
        print '%30s' % 'Executed multiarray in ', str((end - start) * 1000), 'ms'
        print 'multiarray filled!'
        '''
    

    def controlLoop(self):
            #print('In control loop!')
            #self.s = rospy.Service('rgb_to_label_prob', RGB2LabelProb, self.handle_rgb_to_label_prob)
            #rospy.spin()
            if self.obtain_rgb is True:
                self.obtain_rgb = False
                self.rgb_frame = cv2.resize(self.rgb_frame, (self.input_shape[3], self.input_shape[2]))
                b,g,r = cv2.split(self.rgb_frame)       # get b,g,r
                self.rgb_frame = cv2.merge([r,g,b])     # switch it to rgb

                #plt.figure(1) # resized input simage
                #imgplot = plt.imshow(self.rgb_frame) 
                #plt.show(block=True)
                #input("Press Enter to continue...")

                input_image = self.rgb_frame.transpose((2, 0, 1))
                input_image = np.asarray([input_image])
                while input_image.shape[3] != self.input_shape[3]:
                    input_image = self.rgb_frame.transpose((2, 0, 1))

                print 'begin segnet'
                start = time.time()
                out = self.net.forward_all(data=input_image)
                end = time.time()
                #conv1_1_D
                '''
                for clas in range (1, self.num_class):
                    print 'class', clas, self.net.blobs['conv1_1_D'].data[:,clas,1,1:10]
                
                input("Press Enter to continue...")
                '''
                print 'finish segnet'
                print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'
                print 'filling in class prob!'
                start = time.time()

                '''
                for row in range(0, self.height):
                    for col in range(0, self.width):
                        for cl in range (0,self.num_class):
                            self.class_response.data[self.dstride1*row + self.dstride2*col + cl] = self.net.blobs['conv1_1_D'].data[:,cl+1,row,col]            
                '''

                # Global range normalization
                segnet_prob_out = self.net.blobs['conv1_1_D'].data.squeeze()
                segnet_prob_out = segnet_prob_out[1:, :, :]
                segnet_prob_out = np.transpose(segnet_prob_out, axes=(1,2,0))

                segnet_prob_out_max = np.amax(segnet_prob_out)
                segnet_prob_out_min = np.amin(segnet_prob_out)
                segnet_prob_out = (segnet_prob_out-segnet_prob_out_min)/(segnet_prob_out_max - segnet_prob_out_min)

                # Sum to 1 prob distribution normalization
                segnet_pix_sum = np.sum(segnet_prob_out, axis=2)
                for cl in range(0, self.num_class):
                    segnet_prob_out[:,:,cl] = np.divide(segnet_prob_out[:,:,cl], segnet_pix_sum)


                segnet_reshaped_prob_out = np.zeros((self.height, self.width, self.num_class))
                for cl in range(0, self.num_class):
                    segnet_reshaped_prob_out[:,:,cl] = cv2.resize(segnet_prob_out[:,:,cl], (self.width, self.height), interpolation=cv2.INTER_NEAREST)

                # Reshape and make msg
                self.class_response.data = np.reshape(segnet_reshaped_prob_out, segnet_reshaped_prob_out.size, 'C')

                '''
                for row in range(0, self.height):
                    for col in range(0, self.width):
                        for cl in range(0, self.num_class):
                            self.class_response.data[self.dstride1 * row + self.dstride2 * col + cl] = \
                                segnet_reshaped_prob_out[row, col, cl]
                '''

                end = time.time()
                print '%30s' % 'Executed multiarray in ', str((end - start) * 1000), 'ms'
                print 'multiarray filled!'
                self.wait_for_segnet = False




if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(0) # set gpu device

    rospy.init_node( 'semanticRGB_server', log_level=rospy.INFO)

    seg_node = SegnetSemantic()

    seg_node.start()
    rospy.spin()
