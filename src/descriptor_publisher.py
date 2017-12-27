#!/usr/bin/env python

import rospy
import scipy as sp
import csv
from rp_semantic.msg import BoWPDescriptors

def pub_msg(pub, bow, bowp):
    msg1 = BoWPDescriptors()
    msg1.bow = bow
    msg1.bowp = bowp

    print("Sending msg...")
    pub.publish(msg1)
    rospy.sleep(0.5)

if __name__ == '__main__':
    rospy.init_node("bowp_publisher")

    bowp_pub = rospy.Publisher("rp_semantic/place_descriptors", BoWPDescriptors, queue_size=5)
    rospy.sleep(1.0)

    with open('/home/albert/Desktop/room_test/descriptors.txt', 'rb') as csvfile:
        desc_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)

        all_descriptors = list()
        for row in desc_reader:
            all_descriptors.append(row)

    for i in range(0, len(all_descriptors), 2):
        pub_msg(bowp_pub, all_descriptors[i], all_descriptors[i+1])


rospy.sleep(0.5)