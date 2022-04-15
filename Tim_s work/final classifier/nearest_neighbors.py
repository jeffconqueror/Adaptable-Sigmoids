import time
import random

import numpy as np
from sklearn.neighbors import KDTree, BallTree


class LinearNNSearch:
    def __init__(self, data: np.ndarray, fit: bool = False):
        self.data = data
        self.fit = fit

    def nn_distance(self, point: np.ndarray) -> float:
        '''
        point: the point to find nearest neighbor distance to, as a numpy array of features (coordinates in feature space)

        returns: the distance from point to its nearest neighbor in the data, as a float
        '''

        square_diffs = (data - point)**2
        distances = np.sqrt(square_diffs.sum(axis=1))

        if self.fit:
            distances = distances[distances != 0]

        return np.partition(distances, 0)[0]


class TreeNNSearch:
    def __init__(self, data: np.ndarray, tree: 'Tree', fit: bool):
        # TODO: could also configure the tree?
        self.tree = tree(data)
        self.fit = fit

    def nn_distance(self, point: np.ndarray) -> float:
        k = 2 if self.fit else 1
        dist, _ = self.tree.query(np.array([point]), k=k)

        if k == 1:
            return dist[0][0]
        else:
            return dist[0][1]

    def kd_tree(data: np.ndarray, fit: bool = False) -> 'TreeNNSearch':
        return TreeNNSearch(data, KDTree, fit)

    def ball_tree(data: np.ndarray, fit: bool = False) -> 'TreeNNSearch':
        return TreeNNSearch(data, BallTree, fit)


class NearestNeighborMethod:
    LINEAR = LinearNNSearch
    KD_TREE = TreeNNSearch.kd_tree
    BALL_TREE = TreeNNSearch.ball_tree


if __name__ == '__main__':
    # sanity check & benchmarking
    n_pts = 1000000
    n_queries = 1000
    n_features = 20

    print(f'Benchmarking NNS for {n_pts} training points, {n_features} features, & {n_queries} queries')
    print()

    data = np.random.rand(n_pts, n_features) # 10k points with 10 features
    points = np.random.rand(n_queries, n_features) # one random test point

    print('Linear Search')

    start = time.monotonic()
    linear = LinearNNSearch(data)

    for point in points:
        _ = linear.nn_distance(point)

    end = time.monotonic()

    print(f'time elapsed: {end - start} seconds')
    print()


    print('KD Tree')

    start = time.monotonic()
    kd_tree = TreeNNSearch.kd_tree(data)

    for point in points:
        nnd = kd_tree.nn_distance(point)

    end = time.monotonic()

    print(f'time elapsed: {end - start} seconds')
    print()


    print('Ball Tree')

    start = time.monotonic()
    ball_tree = TreeNNSearch.ball_tree(data)

    for point in points:
        nnd = ball_tree.nn_distance(point)

    end = time.monotonic()

    print(f'time elapsed: {end - start} seconds')
