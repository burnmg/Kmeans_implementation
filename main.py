from LoyldMethod import LoyldMethod
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from eigen_tranform import eigen_transform

"""
Following methods are a set of tests. Each test will print a plot about the clustering result. 
If you would like to run a test, simply call the method. 
"""

def test_lolyds_circle_data():
    model = LoyldMethod()

    data, _ = datasets.make_circles(n_samples=100, factor=0.5, noise=.05)
    # print(data)
    # generate data
    model.clustering(data, k=2, iters=100)
    # plot centroids
    axis = list(map(lambda x: x * 2, [-1, 1, -1, 1]))
    plt.axis(axis)
    plt.plot(data[model.clusters[0], 0], data[model.clusters[0], 1], 'bo')
    plt.plot(data[model.clusters[1], 0], data[model.clusters[1], 1], 'ro')
    plt.plot(model.centroids[:,0], model.centroids[:,1], 'y*', markersize=10)
    plt.show()


def test_lolyds_moon_data():
    model = LoyldMethod()

    data, _ = datasets.make_moons(n_samples=100, noise=.05)
    # print(data)
    # generate data
    model.clustering(data, k=2, iters=100)
    # plot centroids
    axis = [-2, 3, -3, 3]
    plt.axis(axis)
    plt.plot(data[model.clusters[0], 0], data[model.clusters[0], 1], 'bo')
    plt.plot(data[model.clusters[1], 0], data[model.clusters[1], 1], 'ro')
    plt.plot(model.centroids[:,0], model.centroids[:,1], 'y*', markersize=10)
    plt.show()


# blobs is gaussian
# blobs is gaussian
def test_lolyds_moon_and_blobs():
    model = LoyldMethod()

    data1, _ = datasets.make_moons(n_samples=100, noise=.05, shuffle=False)
    data2, _ = datasets.make_blobs(n_samples=50, centers=[[0, 0]], cluster_std=0.1, n_features=2, shuffle=False)

    data = np.concatenate((data1[:50, :], data2[:50, :]), axis=0)
    # print(data)
    # generate data
    model.clustering(data, k=2, iters=100)
    # plot centroids
    axis = [-3, 3, -3, 3]
    plt.axis(axis)
    plt.plot(data[model.clusters[0], 0], data[model.clusters[0], 1], 'bo')
    plt.plot(data[model.clusters[1], 0], data[model.clusters[1], 1], 'ro')
    plt.plot(model.centroids[:,0], model.centroids[:,1], 'y*', markersize=10)
    plt.show()

"""
Below codes are eigenvector version's k-means
"""


def test_eigenlolyds_circle_data(r=5):
    model = LoyldMethod()

    data, _ = datasets.make_circles(n_samples=100, factor=0.5, noise=.05)
    transformed_data = eigen_transform(data, r=r, k=2)
    # print(data)
    # generate data
    model.clustering(transformed_data, k=2, iters=100)
    # plot centroids
    axis = list(map(lambda x: x * 2, [-1, 1, -1, 1]))
    plt.axis(axis)
    plt.plot(data[model.clusters[0], 0], data[model.clusters[0], 1], 'bo')
    plt.plot(data[model.clusters[1], 0], data[model.clusters[1], 1], 'ro')
    plt.plot(model.centroids[:, 0], model.centroids[:, 1], 'y*', markersize=10)
    plt.show()


def test_eogenlolyds_moon_and_blobs():
    model = LoyldMethod()

    data1, _ = datasets.make_moons(n_samples=100, noise=.05, shuffle=False)
    data2, _ = datasets.make_blobs(n_samples=50, centers=[[0, 0]], cluster_std=0.1, n_features=2, shuffle=False)

    data = np.concatenate((data1[:50, :], data2[:50, :]), axis=0)
    transformed_data = eigen_transform(data, r=5, k=2)
    # print(data)
    # generate data
    model.clustering(transformed_data, k=2, iters=100)
    # plot centroids
    axis = [-3, 3, -3, 3]
    plt.axis(axis)
    plt.plot(data[model.clusters[0], 0], data[model.clusters[0], 1], 'bo')
    plt.plot(data[model.clusters[1], 0], data[model.clusters[1], 1], 'ro')
    plt.plot(model.centroids[:,0], model.centroids[:,1], 'y*', markersize=10)
    plt.show()

test_eigenlolyds_circle_data(5)