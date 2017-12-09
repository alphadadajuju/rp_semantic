#!/usr/bin/env python

import rospy
import scipy as sp
from rp_semantic.msg import BoWPDescriptors

def pub_msg(pub, bow, bowp):
    msg1 = BoWPDescriptors()
    msg1.bow = bow
    msg1.bowp = bowp

    print("Sending msg...")
    pub.publish(msg1)
    rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node("bowp_publisher")

    bowp_pub = rospy.Publisher("rp_semantic/place_descriptors", BoWPDescriptors, queue_size=5)
    rospy.sleep(1.0)
    print("Publisher initialized")

    pub_msg(bowp_pub, [1000,1000,1000], [1000,1000,1000])

    for i in range(1):
        pub_msg(bowp_pub, 2000*sp.rand(3), 2000*sp.rand(3))

    pub_msg(bowp_pub, [999,1001,995], [999,1001,899])

    for i in range(1):
        pub_msg(bowp_pub, 2000*sp.rand(3), 2000*sp.rand(3))

    pub_msg(bowp_pub, [1000, 1000, 1000], [1000, 1000, 1000])

rospy.sleep(0.5)