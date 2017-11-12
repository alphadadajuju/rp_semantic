#!/usr/bin/env python
# license removed for brevity

import os
import cv2
from threading import Thread, Lock
import rospy
import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import cosine as cosine_dist
from cv_bridge import CvBridge, CvBridgeError

from rp_semantic.msg import BoWP


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2.transpose())
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

class BoWPmatching:
    def __init__(self):
        self.mutex = Lock()
        ### I/O
        self.bowp_subscriber = rospy.Subscriber("rp_semantic/bow_bowp_descriptors", BoWP, self.descriptor_callback)

        self.base_path_output = os.path.expanduser('~/rp_semantic/') + str(rospy.Time.now().secs) + '_loop'
        if not os.path.exists(self.base_path_output):
            os.makedirs(self.base_path_output)

        self.bridge = CvBridge()

        ### CONFIGURABLE PARAMETERS
        self.p = int(3)  # Frame margin for matching min of 6
        self.DEBUG_MODE = False
        self.OUTPUT_MODE = 'file'
        self.dist_exponent = 1.0
        self.posterior_thresh = 0.5
        self.prior_neighbourhood = 2

        ### Operation variables
        self.t = -1 # Timestep indication
        self.previous_loop_closure_event = -1 # Bayesian filtering for temporal coherence

        ### Important variables
        self.last_message = None
        self.node_images = list([])

        # Descriptors are stored as (N_nodes x N_words) matrices
        self.bow_descriptors_list = []
        self.bowp_descriptors_list = []
        self.bow_descriptors = np.matrix([])
        self.bowp_descriptors = np.matrix([])

        print ('BoWP matching initialized:')


    def descriptor_callback(self, msg):
        self.mutex.acquire()
        self.last_message = msg

        # Todo modify message
        if False and self.bow_descriptors is not None and msg.node_id >= 0:
             # Update case
             pass
        else:
            # Additions
            print("Received descriptors")

            self.t += 1

            # Store image associated with this node
            '''
            try:
                img = self.bridge.imgmsg_to_cv2(msg.raw_rgb, "rgb8")
                self.node_images.append(img)
            except CvBridgeError, e:
                rospy.loginfo("Conversion failed")
                rospy.logerr(e)
            '''

            if self.bow_descriptors.size == 0:
                self.bow_descriptors_list = [list(msg.bow)]
                self.bowp_descriptors_list = [list(msg.bowp)]
            else:
                self.bow_descriptors_list.append(msg.bow)
                self.bowp_descriptors_list.append(msg.bowp)

            self.bow_descriptors = np.matrix(self.bow_descriptors_list)
            self.bowp_descriptors = np.matrix(self.bowp_descriptors_list)

            if self.bow_descriptors.shape[0] > self.p+2:
                self.previous_loop_closure_event = self.match_last_frame()

                if self.OUTPUT_MODE is 'file':
                    self.store_node_and_closure(self.base_path_output, self.previous_loop_closure_event)

            if self.previous_loop_closure_event != -1:
                print("Loop closure between " + str(self.t) + " and " + str(self.previous_loop_closure_event))

        self.mutex.release()

    def match_last_frame(self):
        """
        Compute tf_idf weighted descriptors and compute similarity score
        """

        # n_i - count total number of words in each node (sum descriptor rows (axis=1))
        n_i_bow = np.sum(self.bow_descriptors, axis=1)
        n_i_bowp = np.sum(self.bowp_descriptors, axis=1)

        # n_w - count number of words of type w in all nodes (sum descriptor columns (axis=0))
        n_w_bow = np.sum(self.bow_descriptors, axis=0)
        n_w_bowp = np.sum(self.bowp_descriptors, axis=0)

        # N - total number of images seen so far
        N_bow = np.sum(n_i_bow)
        N_bowp = np.sum(n_i_bowp)

        bow_tfidf = np.asmatrix(np.zeros(self.bow_descriptors.shape))
        bowp_tfidf = np.asmatrix(np.zeros(self.bowp_descriptors.shape))

        for i in range(0, self.bow_descriptors.shape[0]):
            bow_tfidf[i] = np.multiply(np.divide(self.bow_descriptors[i], n_i_bow[i]),
                                       np.log( N_bow / (n_w_bow+ 1E-12) ) )
            bowp_tfidf[i] = np.multiply(np.divide(self.bowp_descriptors[i], n_i_bowp[i]),
                                        np.log(N_bowp / (n_w_bowp + 1E-12)) )

        #print("tfidf", bow_tfidf)

        bow_similarity_scores = np.zeros(bow_tfidf.shape[0] - 1 - self.p)
        bowp_similarity_scores = np.zeros(bowp_tfidf.shape[0] - 1 - self.p)
        for i in range(0, bow_tfidf.shape[0] - 1 - self.p):
            bow_similarity_scores[i] = 1.0/np.power(cosine_dist(bow_tfidf[-1], bow_tfidf[i]) + 1E-12, self.dist_exponent)
            bowp_similarity_scores[i] = 1.0/np.power(cosine_dist(bowp_tfidf[-1], bowp_tfidf[i]) + 1E-12, self.dist_exponent)

        if self.DEBUG_MODE:
            print("Similarity scores", bow_similarity_scores)

        ### Compute likelihood

        # Update mean and stddev
        bow_similarity_mean = np.mean(bow_similarity_scores)
        bow_similarity_stddev = np.std(bow_similarity_scores)
        bowp_similarity_mean = np.mean(bowp_similarity_scores)
        bowp_similarity_stddev = np.std(bowp_similarity_scores)

        bow_likelihood = np.ones(bow_similarity_scores.shape[0] + 1)
        bowp_likelihood = np.ones(bow_similarity_scores.shape[0] + 1)
        joint_likelihood = np.zeros(bow_similarity_scores.shape[0] + 1)
        for i in range(0, bow_similarity_scores.shape[0]):
            if bow_similarity_scores[i] > (bow_similarity_mean + bow_similarity_stddev):
                bow_likelihood[i] = (bow_similarity_scores[i] - bow_similarity_stddev) / bow_similarity_mean
            if bowp_similarity_scores[i] > (bowp_similarity_mean + bowp_similarity_stddev):
                bowp_likelihood[i] = (bowp_similarity_scores[i] - bowp_similarity_stddev) / bowp_similarity_mean

            joint_likelihood[i] = bow_likelihood[i]*bowp_likelihood[i]

        # The likelihood of new node is encoded on the final entry
        joint_likelihood[-1] = ((bow_similarity_mean / bow_similarity_stddev) + 1) * \
                                ((bowp_similarity_mean / bowp_similarity_stddev) + 1)

        if(np.isnan(joint_likelihood[-1]) or np.isinf(joint_likelihood[-1])):
            joint_likelihood[-1] = 1

        if self.DEBUG_MODE:
            print("Likelihood", joint_likelihood)

        ### Get posterior
        prior = self.compute_prior(joint_likelihood.shape[0], self.previous_loop_closure_event)
        posterior = np.multiply(joint_likelihood, prior)
        if self.DEBUG_MODE:
            print("Posterior: ", posterior)

        if self.OUTPUT_MODE is 'file':
            self.store_step_file(self.last_message,
                                 bow_similarity_scores, bowp_similarity_scores,
                                 bow_likelihood, bowp_likelihood, joint_likelihood,
                                 prior, posterior)

        return np.argmax(posterior[:-1]) if np.amax(posterior[:-1]) > self.posterior_thresh else -1

    def compute_prior(self, prior_size, previous_loop_closure):
        prior = np.zeros(prior_size)

        if previous_loop_closure == -1:
            prior += 0.1/(self.t - self.p + 1)
            prior[-1] = 0.9
        else:
            neigh_left = np.min([previous_loop_closure, self.prior_neighbourhood])
            neigh_right = np.min([prior_size - previous_loop_closure,  self.prior_neighbourhood])

            gaussian_weights = gaussian(np.arange(-self.prior_neighbourhood, self.prior_neighbourhood, 1),
                                        0, (self.prior_neighbourhood/0.5))
            gaussian_weights = 0.9*(gaussian_weights/np.sum(gaussian_weights))

            gaussian_weights = gaussian_weights[(self.prior_neighbourhood - neigh_left):]

            prior[previous_loop_closure - neigh_left: previous_loop_closure + neigh_right] = gaussian_weights
            prior[-1] = 0.1

        # print("prior", self.prior)
        return prior

    def store_step_file(self, msg_in, bow_sim, bowp_sim, bow_l, bowp_l, joint_l, prior, posterior):
        pass

    def store_node_and_closure(self, base_path, node_closure):
        return

        image_name = str(self.t) + '_' + str(node_closure) + '.jpg'

        if node_closure != -1:
            img = np.concatenate((self.node_images[-1], self.node_images[node_closure]), axis=1)
        else:
            width, height = cv2.cv.GetSize(self.node_images[-1])
            img = np.concatenate((self.node_images[-1], 255 * np.ones((width, height))), axis=1)

        cv2.imwrite(self.base_path_output + image_name, img)

if __name__ == '__main__':
    rospy.init_node("bowp_matching")

    print("Initializing matching")
    np.set_printoptions(suppress=True, precision=5)

    matcher = BoWPmatching()
    rospy.spin()