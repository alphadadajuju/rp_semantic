#!/usr/bin/env python
# license removed for brevity

import numpy as np
import rospy
from std_msgs.msg import String

class BoWPmatching:
    def __init__(self):

        # Frame margin for matching
        self.p = 10 

        ### Important variables

        # Descriptors are stored as (N_nodes x N_words) matrices
        self.bow_descriptors = None
        self.bowp_descriptors = None

        self.bow_tfidf_descriptors = None
        self.bowp_tfidf_descriptors = None

        self.bow_similarity_scores = None
        self.bowp_similarity_scores = None

        # Variables holding mean and stddev for likelihood computation
        self.bow_similarity_mean = None
        self.bow_similarity_stddev = None
        self.bowp_similarity_mean = None
        self.bowp_similarity_stddev = None

    def recompute_similarity_scores_mean_stddev(self):
        pass

    def compute_similarity_scores(self):

        ### Compute tf_idf weighted descriptors

        # n_i - count total number of words in each node (sum descriptor rows (axis=1))
        n_i_bow = np.sum(self.bow_descriptors, axis=1)
        n_i_bowp = np.sum(self.bowp_descriptors, axis=1)

        # n_w - count total number of words w in all nodes (sum descriptor columns (axis=0))
        n_w_bow = np.sum(self.bow_descriptors, axis=0)
        n_w_bowp = np.sum(self.bowp_descriptors, axis=0)

        # N - sum n_i vector
        N_bow = np.sum(n_i_bow)
        N_bowp = np.sum(n_i_bowp)

        bow_tfidf = np.zeros(self.bow_descriptors.shape)
        for i in range(0, self.bow_descriptors.shape[0]):
            bow_tfidf[i] = (self.bow_descriptors[i]/n_i_bow[i]) * np.log((1.0/N_bow) * n_w_bow)

        bowp_tfidf = np.zeros(self.bowp_descriptors.shape)
        for i in range(0, self.bowp_descriptors.shape[0]):
            bowp_tfidf[i] = (self.bowp_descriptors[i]/n_i_bowp[i]) * np.log((1.0/N_bowp) * n_w_bowp)

    def compute_likelihood(self):
        pass




if __name__ == '__main__':

    try:
        talker()
    except rospy.ROSInterruptException:
        pass