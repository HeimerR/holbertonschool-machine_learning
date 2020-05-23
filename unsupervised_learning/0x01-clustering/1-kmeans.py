#!/usr/bin/env python3
"""Perform k means on some data"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform k means on some data
    X: (n, d) ndarray n = data, d = dimensions
    k: number of clusters
    """
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    assignments = np.zeros(X.shape[0], dtype=int)
    while iterations > 0:
        prevassign = assignments.copy()
        update_assignments(centroids, X, assignments)
        for cidx in range(len(centroids)):
            assigned = np.where(assignments == cidx, True, False)
            if assigned.sum() == 0:
                centroids[cidx] = np.random.uniform(np.amin(X, axis=0),
                                                    np.amax(X, axis=0),
                                                    (1, X.shape[1]))
                continue
            centroids[cidx] = X[assigned, :].sum(axis=0) / assigned.sum()
        if (assignments == prevassign).all():
            break
        iterations -= 1
    update_assignments(centroids, X, assignments)
    return centroids, assignments


def update_assignments(centroids, X, assignments):
    """
    Update point assignments to centroids
    """
    for cidx, centroid in enumerate(centroids):
        for pidx, point in enumerate(X):
            assigned = assignments[pidx]
            if assigned == cidx:
                continue
            assigned = pow(point - centroids[assigned], 2).sum()
            checking = pow(point - centroid, 2).sum()
            if checking < assigned:
                assignments[pidx] = cidx


def initialize(X, k):
    """Initialize k clusters with mutivariate uniform distribution"""
    if ((not isinstance(X, np.ndarray) or len(X.shape) != 2 or
         not isinstance(k, int) or k < 1 or k > X.shape[0] or X.shape[1] < 1)):
        return None
    return np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                             (k, X.shape[1]))
