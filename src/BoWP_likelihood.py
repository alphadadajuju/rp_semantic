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


class BoWPmatching:
    def __init__(self):
        self.mutex = Lock()
        ### I/O
        self.bowp_subscriber = rospy.Subscriber("rp_semantic/bow_bowp_descriptors", BoWP, self.descriptor_callback)
        self.bridge = CvBridge()

        ### Operation variables
        self.t = -1 # Timestep indication
        self.previous_loop_closure_event = -1 # Bayesian filtering for temporal coherence
        self.dist_exponent = 1.0

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

        print("Received descriptors")

        if self.bow_descriptors.size == 0:
            self.bow_descriptors_list = [list(msg.bow)]
            self.bowp_descriptors_list = [list(msg.bowp)]
        else:
            self.bow_descriptors_list.append(msg.bow)
            self.bowp_descriptors_list.append(msg.bowp)

        self.bow_descriptors = np.matrix(self.bow_descriptors_list)
        self.bowp_descriptors = np.matrix(self.bowp_descriptors_list)


        if self.bow_descriptors.shape[0] > 1:
            self.likelihood_matrix = self.likelihood_between_descriptors()

        self.mutex.release()

    def similarity_between_descriptors(self):
        # n_i - count total number of words in each node (sum descriptor rows (axis=1))
        n_i_bow = np.sum(self.bow_descriptors, axis=1)
        n_i_bowp = np.sum(self.bowp_descriptors, axis=1)

        # n_w - count number of words of type w in all nodes (sum descriptor columns (axis=0))
        n_w_bow = np.sum(self.bow_descriptors, axis=0)
        n_w_bowp = np.sum(self.bowp_descriptors, axis=0)

        # N - total number of images seen so far
        N_bow = np.sum(n_i_bow)
        N_bowp = np.sum(n_i_bowp)

        ### For each descriptor, compute the likelihoods
        bow_tfidf = np.asmatrix(np.zeros(self.bow_descriptors.shape))
        bowp_tfidf = np.asmatrix(np.zeros(self.bowp_descriptors.shape))

        for i in range(0, self.bow_descriptors.shape[0]):
            bow_tfidf[i] = np.multiply(np.divide(self.bow_descriptors[i], n_i_bow[i]),
                                       np.log(N_bow / (n_w_bow + 1E-12)))
            bowp_tfidf[i] = np.multiply(np.divide(self.bowp_descriptors[i], n_i_bowp[i]),
                                        np.log(N_bowp / (n_w_bowp + 1E-12)))

        bow_similarity_scores = np.zeros(bow_tfidf.shape[0] - 1)
        bowp_similarity_scores = np.zeros(bowp_tfidf.shape[0] - 1)
        for i in range(0, bow_tfidf.shape[0] - 1):
            bow_similarity_scores[i] = 1.0 / np.power(cosine_dist(bow_tfidf[-1], bow_tfidf[i]) + 1E-12,
                                                      self.dist_exponent)
            bowp_similarity_scores[i] = 1.0 / np.power(cosine_dist(bowp_tfidf[-1], bowp_tfidf[i]) + 1E-12,
                                                       self.dist_exponent)

        # TODO compute for each descriptor with every other

    def likelihood_between_descriptors(self):
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


        ### For each descriptor, compute the likelihoods
        bow_tfidf = np.asmatrix(np.zeros(self.bow_descriptors.shape))
        bowp_tfidf = np.asmatrix(np.zeros(self.bowp_descriptors.shape))

        for i in range(0, self.bow_descriptors.shape[0]):
            bow_tfidf[i] = np.multiply(np.divide(self.bow_descriptors[i], n_i_bow[i]),
                                       np.log(N_bow / (n_w_bow + 1E-12)))
            bowp_tfidf[i] = np.multiply(np.divide(self.bowp_descriptors[i], n_i_bowp[i]),
                                        np.log(N_bowp / (n_w_bowp + 1E-12)))

        bow_similarity_scores = np.zeros(bow_tfidf.shape[0] - 1)
        bowp_similarity_scores = np.zeros(bowp_tfidf.shape[0] - 1)
        for i in range(0, bow_tfidf.shape[0] - 1):
            bow_similarity_scores[i] = 1.0/np.power(cosine_dist(bow_tfidf[-1], bow_tfidf[i]) + 1E-12, self.dist_exponent)
            bowp_similarity_scores[i] = 1.0/np.power(cosine_dist(bowp_tfidf[-1], bowp_tfidf[i]) + 1E-12, self.dist_exponent)

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

if __name__ == '__main__':
    rospy.init_node("bowp_matching")

    print("Initializing matching")
    np.set_printoptions(suppress=True, precision=5)

    matcher = BoWPmatching()
    rospy.spin()
