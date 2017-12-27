#!/usr/bin/env python

import rospy
import scipy as sp
import csv
from rp_semantic.msg import BoWPDescriptors


class DescriptorSaver:
    def __init__(self):
        self.descriptor_file_path = rospy.get_param("rp_semantic/descriptor_file", False)

        desc_sub = rospy.Subscriber("rp_semantic/place_descriptors", BoWPDescriptors, self.append_descriptor, queue_size=5)

    def append_descriptor(self, msg):

        with open(self.descriptor_file_path, 'a+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

            writer.writerow(msg.bow)
            writer.writerow(msg.bowp)


if __name__ == '__main__':
    rospy.init_node("descriptor_storage")

    desc_saver = DescriptorSaver()

    rospy.spin()





