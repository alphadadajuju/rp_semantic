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

from rp_semantic.msg import BoWP, BoWPDescriptors


class BowpPlaceMatching:
    def __init__(self):
        self.mutex = Lock()
        ### I/O
        self.bowp_subscriber = rospy.Subscriber("rp_semantic/place_descriptors", BoWPDescriptors, self.descriptor_callback)
        self.bridge = CvBridge()

        ### Operation variables
        self.dist_exponent = 3.0

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

        # Remove wall, ceiling and floor labels
        self.bow_descriptors[-1, 0] = self.bow_descriptors[-1, 1] = self.bow_descriptors[-1, 21] = 0

        if self.bow_descriptors.shape[0] > 1:
            scores = self.similarity_between_descriptors()
            print("Similarity based on bow")
            print(scores[0])
            print("Similarity based on bowp")
            print(scores[1])
            print("Similarity based on both")
            print(scores[2])

        self.mutex.release()

    def similarity_between_descriptors(self):

        # But first, compute tfidf weighted vectors
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

        # Initialize similarity matrix, that says prob of roomA being roomB sim_mat[roomA][roomB]
        bow_similarity_scores = np.zeros((self.bow_descriptors.shape[0], self.bow_descriptors.shape[0]))
        bowp_similarity_scores = np.zeros((self.bow_descriptors.shape[0], self.bow_descriptors.shape[0]))

        for i in range(0, bow_tfidf.shape[0]):
            for j in range(0, bow_tfidf.shape[0]):
                if i == j:
                    bow_similarity_scores[i][j] = 1.0
                    bowp_similarity_scores[i][j] = 1.0
                else:
                    bow_similarity_scores[i][j] = 1.0/np.power(cosine_dist(bow_tfidf[i], bow_tfidf[j]) + np.finfo(float).eps, self.dist_exponent)
                    bowp_similarity_scores[i][j] = 1.0/np.power(cosine_dist(bowp_tfidf[i], bowp_tfidf[j]) + np.finfo(float).eps, self.dist_exponent)


        bow_likelihood = np.ones((self.bow_descriptors.shape[0], self.bow_descriptors.shape[0]))
        bowp_likelihood = np.ones((self.bow_descriptors.shape[0], self.bow_descriptors.shape[0]))

        for i in range(0, bow_similarity_scores.shape[0]):
            bow_sim_scores = np.array(bow_similarity_scores[i])
            bowp_sim_scores = np.array(bowp_similarity_scores[i])

            # Remove self scores
            bow_sim_scores_temp = np.delete(bow_sim_scores, i)
            bowp_sim_scores_temp = np.delete(bowp_sim_scores, i)

            # Update mean and stddev
            bow_similarity_mean = np.mean(bow_sim_scores_temp)
            bow_similarity_stddev = np.std(bow_sim_scores_temp)
            bowp_similarity_mean = np.mean(bowp_sim_scores_temp)
            bowp_similarity_stddev = np.std(bowp_sim_scores_temp)

            #Compute new likelihood
            for j in range(bow_sim_scores.size):
                if bow_sim_scores[j] > (bow_similarity_mean + bow_similarity_stddev):
                    bow_likelihood[i][j] = (bow_sim_scores[j] - bow_similarity_stddev) / bow_similarity_mean

                if bowp_sim_scores[j] > (bowp_similarity_mean + bowp_similarity_stddev):
                    bowp_likelihood[i][j] = (bowp_sim_scores[j] - bowp_similarity_stddev) / bowp_similarity_mean

        joint_likelihood = np.multiply(bow_likelihood, bowp_likelihood)

        return (bow_likelihood, bowp_likelihood, joint_likelihood)

if __name__ == '__main__':
    rospy.init_node("bowp_place_matching")

    print("Initializing matching")
    np.set_printoptions(suppress=True, precision=4)

    matcher = BowpPlaceMatching()
    rospy.spin()
