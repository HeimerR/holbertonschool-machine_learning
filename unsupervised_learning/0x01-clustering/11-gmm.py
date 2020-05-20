#!/usr/bin/env python3
""" GMM """
import sklearn.mixture


def gmm(X, k):
    """
     calculates a GMM from a dataset:

        - X is a numpy.ndarray of shape (n, d) containing the dataset
        - k is the number of clusters

        The only import you are allowed to use is import sklearn.mixture

        Returns: pi, m, S, clss, bic
        - pi is a numpy.ndarray of shape (k,) containing the cluster priors
        - m is a numpy.ndarray of shape (k, d) containing the centroid means
        - S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices
        - clss is a numpy.ndarray of shape (n,) containing the cluster indices
            for each data point
        - bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    G = GMM.fit(X)
    L = GMM.fit_predict(X)
    bic = GMM.bic(X)

    return G.weights_, G.means_, G.covariances_, L, bic
