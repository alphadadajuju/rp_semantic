#!/usr/bin/env python
# license removed for brevity

import numpy as np
from scipy.spatial.distance import cosine as cosine_between
import rospy
from rp_semantic.msg import BoWP
from threading import Thread, Lock

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class BoWPmatching:
    def __init__(self):
        self.mutex = Lock()
        # I/O
        self.bowp_subscriber = rospy.Subscriber("rp_semantic/bow_bowp_descriptors", BoWP, self.descriptor_callback)

        # Timestep indication
        self.t = -1

        # Frame margin for matching
        self.p = 1

        ### Important variables
        # Descriptors are stored as (N_nodes x N_words) matrices
        self.bow_descriptors = None
        self.bowp_descriptors = None

        self.bow_similarity_scores = None
        self.bowp_similarity_scores = None

        # Variables holding mean and stddev for likelihood computation
        self.bow_similarity_mean = None
        self.bow_similarity_stddev = None
        self.bowp_similarity_mean = None
        self.bowp_similarity_stddev = None

        self.joint_likelihood = None

        # Bayesian filtering for temporal coherence
        self.prior_neighbourhood = 3

        self.previous_loop_closure_event = -1
        self.prior = None

        self.posterior = None


    def descriptor_callback(self, msg):
        self.mutex.acquire()

        # TODO implement node_id update
        # if self.bow_descriptors is not None and msg.node_id in range(self.bow_descriptors.shape[0]-1):
        #     # Update case
        #     pass
        # else:

        # Additions
        print("Received new descriptor")
        self.t += 1
        if self.bow_descriptors is None:
            self.bow_descriptors = np.matrix(msg.bow)
            self.bowp_descriptors = np.matrix(msg.bowp)
        else:
            self.bow_descriptors = np.append(self.bow_descriptors, np.matrix(msg.bow), axis=0)
            self.bowp_descriptors = np.append(self.bowp_descriptors, np.matrix(msg.bowp), axis=0)

            if self.bow_descriptors.shape[0] <= self.p:
                self.mutex.release()
                return

            self.previous_loop_closure_event = self.match_last_frame()

        print("Loop closure at time " + str(self.t) + " is " + str(self.previous_loop_closure_event))

        self.mutex.release()

    def match_last_frame(self):
        self.compute_similarity_scores()
        self.compute_likelihood()
        self.compute_prior(self.previous_loop_closure_event)
        self.posterior = np.multiply(self.joint_likelihood, self.prior)

        return np.argmax(self.posterior[:-1]) if np.amax(self.posterior[:-1]) > 0.5 else -1


    def compute_similarity_scores(self):
        """
         Compute tf_idf weighted descriptors and compute similarity score
        """

        # n_i - count total number of words in each node (sum descriptor rows (axis=1))
        n_i_bow = np.sum(self.bow_descriptors, axis=1)
        n_i_bowp = np.sum(self.bowp_descriptors, axis=1)

        # n_w - count total number of words w in all nodes (sum descriptor columns (axis=0))
        n_w_bow = np.sum(self.bow_descriptors, axis=0)
        n_w_bowp = np.sum(self.bowp_descriptors, axis=0)

        # N - sum n_i vector
        N_bow = np.sum(n_i_bow)
        N_bowp = np.sum(n_i_bowp)

        bow_tfidf = np.asmatrix(np.zeros(self.bow_descriptors.shape))
        bowp_tfidf = np.asmatrix(np.zeros(self.bowp_descriptors.shape))
        for i in range(0, self.bow_descriptors.shape[0]):
            bow_tfidf[i] = np.multiply(np.divide(self.bow_descriptors[i],n_i_bow[i]),
                                       np.log((1.0/N_bow) * n_w_bow + 1E-9))
            bowp_tfidf[i] = np.multiply(np.divide(self.bowp_descriptors[i], n_i_bowp[i]),
                                        np.log((1.0 / N_bowp) * n_w_bowp + 1E-9))

        self.bow_similarity_scores = np.zeros(bow_tfidf.shape[0]-1)
        self.bowp_similarity_scores = np.zeros(bowp_tfidf.shape[0]-1)
        for i in range(0, bow_tfidf.shape[0]-1):
            self.bow_similarity_scores[i] = cosine_between(bow_tfidf[-1], bow_tfidf[i])
            self.bowp_similarity_scores[i] = cosine_between(bowp_tfidf[-1], bowp_tfidf[i])

    def compute_similarity_scores_mean_stddev(self):
        self.bow_similarity_mean = np.mean(self.bow_similarity_scores)
        self.bow_similarity_stddev = np.std(self.bow_similarity_scores) + 1E-9
        self.bowp_similarity_mean = np.mean(self.bowp_similarity_scores)
        self.bowp_similarity_stddev = np.std(self.bowp_similarity_scores) + 1E-9

        #print(self.bow_similarity_mean, self.bow_similarity_stddev, self.bowp_similarity_mean, self.bowp_similarity_stddev)

    def compute_likelihood(self):
        # Update mean and stddev
        self.compute_similarity_scores_mean_stddev()

        self.joint_likelihood = np.zeros(self.bow_similarity_scores.shape[0] + 1)
        for i in range(0, self.bow_similarity_scores.shape[0]):
            bow_likelihood, bowp_likelihood = 1, 1
            if self.bow_similarity_scores[i] >= (self.bow_similarity_mean + self.bow_similarity_stddev):
                bow_likelihood = (self.bow_similarity_scores[i] - self.bow_similarity_stddev) / self.bow_similarity_mean
            if self.bowp_similarity_scores[i] >= (self.bowp_similarity_mean + self.bowp_similarity_stddev):
                bowp_likelihood = (self.bowp_similarity_scores[i] - self.bowp_similarity_stddev) / self.bowp_similarity_mean

            self.joint_likelihood[i] = bow_likelihood*bowp_likelihood

        # The likelihood of new node is encoded on the final entry
        self.joint_likelihood[-1] = ((self.bow_similarity_mean / self.bow_similarity_stddev) + 1) * \
                                    ((self.bowp_similarity_mean / self.bowp_similarity_stddev) + 1)
        #print(self.joint_likelihood)

    def compute_prior(self, previous_loop_closure):
        self.prior = np.zeros(self.joint_likelihood.shape[0])

        if previous_loop_closure == -1:
            self.prior += 0.1/(self.t - self.p + 1)
            self.prior[-1] = 0.9
        else:
            gaussian_weights = gaussian(np.arange(-self.prior_neighbourhood, self.prior_neighbourhood, 1),
                                        0, (self.prior_neighbourhood/2.0))
            gaussian_weights = 0.9*(gaussian_weights/np.sum(gaussian_weights))

            self.prior[previous_loop_closure - self.prior_neighbourhood:
                       previous_loop_closure + self.prior_neighbourhood] = gaussian_weights
            self.prior[-1] = 0.1

        #print("prior", self.prior)

if __name__ == '__main__':
    rospy.init_node("bowp_matching")

    matcher = BoWPmatching()
    rospy.spin()