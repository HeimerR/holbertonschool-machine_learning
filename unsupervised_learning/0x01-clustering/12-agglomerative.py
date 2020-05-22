#!/usr/bin/env python3
""" Agglomerative """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """  agglomerative clustering on a dataset:

        - X is a numpy.ndarray of shape (n, d) containing the dataset
        - dist is the maximum cophenetic distance for all clusters

        Performs agglomerative clustering with Ward linkage
        Displays the dendrogram with each cluster displayed
            in a different color

        Returns: clss, a numpy.ndarray of shape (n,) containing
            the cluster indices for each data point
    """
    h = scipy.cluster.hierarchy
    Z = h.linkage(X, 'ward')
    ind = h.fcluster(Z, t=dist, criterion="distance")
    fig = plt.figure()
    dn = h.dendrogram(Z, color_threshold=dist)
    plt.show()
    return ind
