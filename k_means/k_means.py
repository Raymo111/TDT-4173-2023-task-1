import numpy as np
import pandas as pd


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, n_clusters=2):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.k = n_clusters
        self.centroids = None

    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        n_samples = X.shape[0]  # of samples

        a = X.to_numpy()

        # Initialize centroids - spread out
        centroid_indices = np.random.choice(n_samples, 1, replace=False)
        self.centroids = [a[centroid_indices[0]]]
        for _ in range(self.k - 1):
            distances = np.min([euclidean_distance(a, c) for c in self.centroids], axis=0)
            probabilities = distances / np.sum(distances)
            new_centroid_index = np.random.choice(n_samples, p=probabilities)
            self.centroids.append(a[new_centroid_index])
        self.centroids = np.array(self.centroids)

        # Init assign points to clusters
        cluster_assignments = cross_euclidean_distance(a, self.centroids).argmin(axis=1)

        # Update centroids
        while True:
            # Update centroids
            for i in range(self.k):
                self.centroids[i] = np.mean(a[cluster_assignments == i], axis=0)

            # Assign points to clusters
            new_cluster_assignments = cross_euclidean_distance(a, self.centroids).argmin(axis=1)

            # Check if converged
            if np.all(new_cluster_assignments == cluster_assignments):
                break
            else:
                cluster_assignments = new_cluster_assignments

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        a = X.to_numpy()
        return np.array(cross_euclidean_distance(a, self.centroids).argmin(axis=1))

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model = KMeans()
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Computes the cross Euclidean distance between two sets of points.

    Args:
        x, y (array<..., n>): float tensors with pairs of n-dimensional points.
            If y is None, the function computes the Euclidean distance between
            points in x against itself.

    Returns:
        A float array of shape <...> with the pairwise distances between points
        in x and points in y (or within x if y is None).
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum(axis=1)

    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
