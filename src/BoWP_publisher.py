#!/usr/bin/env python

import rospy
import scipy as sp
from rp_semantic.msg import BoWP

def pub_msg(pub, id, bow, bowp):
    msg1 = BoWP()
    msg1.node_id = id
    msg1.bow = bow
    msg1.bowp = bowp

    print("Sending msg...")
    pub.publish(msg1)
    rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node("bowp_publisher")

    bowp_pub = rospy.Publisher("rp_semantic/bow_bowp_descriptors", BoWP, queue_size=5)
    rospy.sleep(1.0)

    pub_msg(bowp_pub, 0, [1000,1000,1000], [1000,1000,1000])

    for i in range(10):
        pub_msg(bowp_pub, 0, 2000*sp.rand(3), 2000*sp.rand(3))

    pub_msg(bowp_pub, 0, [999,1001,1000], [999,1001,1000])

    for i in range(10):
        pub_msg(bowp_pub, 0, 2000*sp.rand(3), 2000*sp.rand(3))

    pub_msg(bowp_pub, 0, [1000, 1000, 1000], [1000, 1000, 1000])

    rospy.sleep(0.5)


