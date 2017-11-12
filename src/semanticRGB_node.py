#!/usr/bin/env python
# ROS imports
import roslib; roslib.load_manifest('segnet_program')
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
from segnet_program.msg import Frame

class SegnetSemantic:
    def __init__(self):
        print ('Semantic initialized:')

    def start(self):

        self.rosSetup()
        while not rospy.is_shutdown():
            self.controlLoop()
        

    def rosSetup(self):
        # caffe_root = '/home/alpha/github/caffe-segnet-cudnn5/'
        caffe_root = '/home/albert/GitHub/caffe-segnet-cudnn5/'

        # model_path = '/home/alpha/catkin_ws/src/segnet_program/src/'
        model_path = '/home/albert/rp_data'

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
        colours_img = cv2.imread(colours)


        if colours_img is None:
            exit()
  
        self.label_colours = colours_img.astype(np.uint8)

        self.bridge = CvBridge() # for decoding sensor_msgs Image data[]
        
        # define class variables
        self.f_height = rospy.get_param("rp_semantic/semanticRGB_node/f_height",0)
        self.f_width = rospy.get_param("rp_semantic/semanticRGB_node/f_height",0) 
        self.wait_for_new_frame = rospy.get_param("rp_semantic/semanticRGB_node/f_height",True)
        self.rgb_has_fresh = rospy.get_param("rp_semantic/semanticRGB_node/f_height",False)
        self.pointcloud_has_fresh = rospy.get_param("rp_semantic/semanticRGB_node/f_height",False)
        self.node_id = rospy.get_param("rp_semantic/semanticRGB_node/f_height",1)
        self.frame_message = Frame()
        self.pointcloud_message = PointCloud2()
        self.rgb_message = Image()
        # 
        # subscriber
        self.rgb_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback, queue_size=1, buff_size=2**24) # change topic's name accordingly
        self.rgb_cam_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.rgb_cam_callback) # change topic's name accordingly
        self.pointcloud_sub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.pointcloud_callback, queue_size=1, buff_size=2**24) # change topic's name accordingly

        # publisher
        self.semantic_pub = rospy.Publisher('/rp_semantic/labels_pointcloud', Frame, queue_size=1) # change topic's name accordingly

        
        
        

    def rgb_callback(self, rgb_msg):
        ''' receive rgb image, resize according to segnet model and feed to segnet; bayer_grbg8 another encoding method? bgr8
        '''
        #print (self.wait_for_new_frame)
        if self.wait_for_new_frame is False:
            return 0
        else:
            try:
        # bgr8 is the pixel encoding -- 8 bits per color, organized as blue/green/red
                self.rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            except CvBridgeError, e:
        # all print statements should use a rospy.log_ form, don't print!
                rospy.loginfo("Conversion failed")
        
            self.rgb_frame = cv2.resize(self.rgb_frame, (self.input_shape[3], self.input_shape[2]))
            self.rgb_has_fresh = True
            self.rgb_msg = rgb_msg
            #print('shape' + str(self.rgb_frame.shape))
            #self.wait_for_new_frame = False

    def pointcloud_callback(self, pointcloud_msg):

        if self.wait_for_new_frame is False:
            return 0
        else:
            self.pointcloud_has_fresh = True
            self.pointcloud_message = pointcloud_msg
            #self.wait_for_new_frame = False



    def rgb_cam_callback(self, rgb_cam_msg):
        ''' receive frame's dimension
        '''
        self.f_height = rgb_cam_msg.height
        self.f_width = rgb_cam_msg.width
        print (self.f_height)
        print (self.f_width)


    def controlLoop(self):
            #print ('rgb:' + str(self.rgb_has_fresh))
            #print('pointcloud:' + str(self.pointcloud_has_fresh))

            if self.rgb_has_fresh is True and self.pointcloud_has_fresh is True: #and self.pointcloud_has_fresh is True:
                print ('enter main control loop')
                self.wait_for_new_frame = False

                #plt.figure(1) # resized input simage
                #imgplot = plt.imshow(self.rgb_frame) 
                #plt.show(block=False)

                input_image = self.rgb_frame.transpose((2, 0, 1))
                input_image = np.asarray([input_image])
                print('shape' + str(input_image.shape))
                # run through Segnet
                if input_image.shape[3] != self.input_shape[3]:
                    self.rgb_has_fresh = False
                    self.pointcloud_has_fresh = False
                    self.wait_for_new_frame = True
                    return 0
                else:
                    start = time.time()
                    out = self.net.forward_all(data=input_image)
                    end = time.time()
                    #print (self.net.blobs['argmax'].data.shape) # 1 x 1 x 224 x 224
                    #print (self.net.blobs['conv1_1_D'].data.shape) #(1, 38, 224, 224); can access probability at 'conv1_1_D' per class per pixel 1 x 38 x 224 x 224
                    #(3, 1, 224, 224) for 'argmax'; out is a 'dict' having only key 'dict'

            
                    segmentation_ind = np.squeeze(self.net.blobs['argmax'].data) # squeeze removes dim = 1 (1x3x24 => 3x24)
                    segmentation_ind_3ch = np.resize(segmentation_ind, (3, self.input_shape[2], self.input_shape[3]))
                    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
                    segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
                    #print (str(segmentation_rgb.shape)) #224 x 224 x 3
                    cv2.LUT(segmentation_ind_3ch, self.label_colours, segmentation_rgb)
                    segmentation_rgb = segmentation_rgb.astype(float) / 255

            
                    print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'

                    #plt.figure() # semantic label
                    #plt.imshow(segmentation_rgb) 
                    #plt.show(block=True)

                    self.frame_message.node_id = self.node_id
                    self.node_id += 1
                    print ('shape of image:' + str(segmentation_ind.shape))
                    self.frame_message.label = self.bridge.cv2_to_imgmsg(np.uint8(segmentation_ind), "mono8")
                    self.frame_message.raw_rgb = self.rgb_message
                    self.frame_message.raw_pointcloud = self.pointcloud_message
                    self.semantic_pub.publish(self.frame_message)

                    # for debug
                    #print ('shape of label message: ' + str(self.frame_message.label))
                    self.test = self.bridge.imgmsg_to_cv2(self.frame_message.label, "mono8")
                    if np.any(self.test > 38):
                        print('bad news')
                        print(self.test)
                    else:
                        print("good news")

                    self.rgb_has_fresh = False
                    self.pointcloud_has_fresh = False
                    self.wait_for_new_frame = True
                    

                    print ('end of main control loop')
                
            
            
            
    

if __name__ == '__main__':
    '''
    sys.path.append('/usr/local/lib/python2.7/site-packages')
    caffe_root = '/home/alpha/github/caffe-segnet-cudnn5/'
    sys.path.insert(0, caffe_root + 'python')
    '''
    caffe.set_mode_gpu()
    caffe.set_device(0) # set gpu device

    rospy.init_node( 'semanticRGB_node', log_level=rospy.INFO)
    '''
    sys.path.append('/usr/local/lib/python2.7/site-packages')
    caffe_root = '/home/alpha/github/caffe-segnet-cudnn5/'
    sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_gpu()
    caffe.set_device(0) # set gpu device
    '''
    seg_node = SegnetSemantic()
    seg_node.start()
