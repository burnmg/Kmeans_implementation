import numpy as np

class LoyldMethod(object):

    def __init__(self):
        self.clusters = None
        self.centroids = None


    def clustering(self, data, k=2, iters=1000, init_centroids=None):

        """
        :param data:  m*n np array. m is data size, n is dimension
        :param k: nums of clusters
        :return:
        """

        # initialize k clusters and centroids
        clusters = None
        m, n = data.shape
        if init_centroids is not None:
            centroids = init_centroids
        else:
            centroids = np.random.rand(k, n)



        # loop start
        iter_count = 0
        while iter_count < iters:
            # assign data to centroids
            new_clusters = [[] for _ in range(k)]
            for i in range(data.shape[0]):
                instance = data[[i], :]
                # find closest centroids
                dists = np.apply_along_axis(lambda x: self.distance(x, instance[0]), 1, centroids)
                c = np.argmin(dists)
                new_clusters[c].append(i)
            # record non-empty clusters



            clusters = new_clusters

            # update centroids
            for i in range(centroids.shape[0]):
                if len(clusters[i]) == 0:
                    draw = np.random.choice(data.shape[0], 1, replace=False)[0]
                    centroids[i] = data[draw]
                    pass
                else:
                    centroids[i] = np.mean(data[clusters[i],:], axis=0)
                    pass
            iter_count += 1

        self.centroids = centroids
        self.clusters = clusters

    def distance(self, x:np.ndarray, y:np.ndarray) -> float:
        return np.linalg.norm(x-y)

