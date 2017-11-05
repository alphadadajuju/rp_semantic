#!/usr/bin/env python

import rospy
from rp_semantic.msg import BoWP

def pub_msg(pub, id, bow, bowp):
    msg1 = BoWP()
    msg1.node_id = id
    msg1.bow = bow
    msg1.bowp = bowp

    print("Sending msg...")
    pub.publish(msg1)
    rospy.sleep(0.5)

if __name__ == '__main__':
    rospy.init_node("bowp_publisher")

    bowp_pub = rospy.Publisher("rp_semantic/bow_bowp_descriptors", BoWP, queue_size=5)


    pub_msg(bowp_pub, 1, [10,10,10], [10,10,10])
    pub_msg(bowp_pub, 2, [1,2,3], [1,2,3])
    pub_msg(bowp_pub, 3, [1,3,3], [1,3,3])
    pub_msg(bowp_pub, 4, [3,2,3], [3,2,3])
    pub_msg(bowp_pub, 5, [10,10,10], [10,10,10])



