#!/usr/bin/env python
# license removed for brevity

import numpy as np
import numpy.linalg as la
import rospy
from rp_semantic.msg import BoWP
from threading import Thread, Lock
from scipy.spatial.distance import cosine as cosine_dist

DEBUG_MODE = False

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
        # I/O
        self.bowp_subscriber = rospy.Subscriber("rp_semantic/bow_bowp_descriptors", BoWP, self.descriptor_callback)

        self.t = -1 # Timestep indication
        self.p = 3 # Frame margin for matching min of 6

        ### Important variables
        # Descriptors are stored as (N_nodes x N_words) matrices
        self.bow_descriptors_list = []
        self.bowp_descriptors_list = []

        self.bow_descriptors = np.matrix([])
        self.bowp_descriptors = np.matrix([])

        # Bayesian filtering for temporal coherence
        self.prior_neighbourhood = 1
        self.previous_loop_closure_event = -1


    def descriptor_callback(self, msg):
        self.mutex.acquire()

        # Todo modify message
        if False and self.bow_descriptors is not None and msg.node_id >= 0:
             # Update case
             pass
        else:
            # Additions
            if DEBUG_MODE:
                print("Received descriptors", msg.bow, msg.bowp)

            self.t += 1

            if self.bow_descriptors.size == 0:
                self.bow_descriptors_list = [msg.bow]
                self.bowp_descriptors_list = [msg.bowp]
            else:
                self.bow_descriptors_list.append(msg.bow)
                self.bowp_descriptors_list.append(msg.bowp)

            self.bow_descriptors = np.matrix(self.bow_descriptors_list)
            self.bowp_descriptors = np.matrix(self.bowp_descriptors_list)

            if self.bow_descriptors.shape[0] > self.p:
                self.previous_loop_closure_event = self.match_last_frame()

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

        # n_w - count number of images containing w in all nodes (sum descriptor columns (axis=0))
        n_w_bow = np.sum(self.bow_descriptors, axis=0)
        n_w_bowp = np.sum(self.bowp_descriptors, axis=0)

        # N - total number of images seen so far
        N_bow = np.sum(n_i_bow)
        N_bowp = np.sum(n_i_bowp)

        bow_tfidf = np.asmatrix(np.zeros(self.bow_descriptors.shape))
        bowp_tfidf = np.asmatrix(np.zeros(self.bowp_descriptors.shape))
        for i in range(0, self.bow_descriptors.shape[0]):
            bow_tfidf[i] = np.multiply(np.divide(self.bow_descriptors[i], n_i_bow[i]),
                                       np.log( N_bow / n_w_bow ))
            bowp_tfidf[i] = np.multiply(np.divide(self.bowp_descriptors[i], n_i_bowp[i]),
                                        np.log(N_bowp / n_w_bowp))

            #Normalize the tf-idf vectors
            #bow_tfidf[i] = bow_tfidf[i]/np.sum(bow_tfidf[i])
            #bowp_tfidf[i] = bow_tfidf[i]/np.sum(bowp_tfidf[i])

        #print("tfidf", bow_tfidf)

        bow_similarity_scores = np.zeros(bow_tfidf.shape[0] - 1)
        bowp_similarity_scores = np.zeros(bowp_tfidf.shape[0] - 1)
        for i in range(0, bow_tfidf.shape[0] - 1):
            bow_similarity_scores[i] = 1.0/(cosine_dist(bow_tfidf[-1], bow_tfidf[i]) + 1E-12)
            bowp_similarity_scores[i] = 1.0/(cosine_dist(bowp_tfidf[-1], bowp_tfidf[i]) + 1E-12)

        if DEBUG_MODE:
            print("Similarity scores", bow_similarity_scores)

        ### Compute likelihood

        # Update mean and stddev
        bow_similarity_mean = np.mean(bow_similarity_scores)
        bow_similarity_stddev = np.std(bow_similarity_scores)
        bowp_similarity_mean = np.mean(bowp_similarity_scores)
        bowp_similarity_stddev = np.std(bowp_similarity_scores)

        joint_likelihood = np.zeros(bow_similarity_scores.shape[0] + 1)
        for i in range(0, bow_similarity_scores.shape[0]):
            bow_likelihood, bowp_likelihood = 1, 1
            if bow_similarity_scores[i] > (bow_similarity_mean + bow_similarity_stddev):
                bow_likelihood = (bow_similarity_scores[i] - bow_similarity_stddev) / bow_similarity_mean
            if bowp_similarity_scores[i] > (bowp_similarity_mean + bowp_similarity_stddev):
                bowp_likelihood = (bowp_similarity_scores[i] - bowp_similarity_stddev) / bowp_similarity_mean

            joint_likelihood[i] = bow_likelihood*bowp_likelihood

        # The likelihood of new node is encoded on the final entry
        joint_likelihood[-1] = ((bow_similarity_mean / bow_similarity_stddev) + 1) * \
                                ((bowp_similarity_mean / bowp_similarity_stddev) + 1)

        if(np.isnan(joint_likelihood[-1]) or np.isinf(joint_likelihood[-1])):
            joint_likelihood[-1] = 1

        if DEBUG_MODE:
            print("Likelihood", joint_likelihood)

        ### Get posterior
        prior = self.compute_prior(joint_likelihood.shape[0], self.previous_loop_closure_event)
        posterior = np.multiply(joint_likelihood, prior)

        if DEBUG_MODE:
            print("Posterior: ", posterior)

        return np.argmax(posterior[:-1]) if np.amax(posterior[:-1]) > 0.5 else -1

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



if __name__ == '__main__':
    rospy.init_node("bowp_matching")

    np.set_printoptions(suppress=True, precision=5)

    matcher = BoWPmatching()
    rospy.spin()