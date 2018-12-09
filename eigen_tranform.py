import numpy as np
def eigen_transform(data, r=1, k=2):
    """

    :param data: ndarray, m*n. m:data_size. n: feature_size
    :return: ndarray. m*k
    """
    # build graph W in m*m matrix
    m = data.shape[0]
    data = data[:, :, None]
    diff = data - data.T
    squared = diff * diff

    dist_matrix = np.sqrt(squared.sum(1))

    top_indices = np.apply_along_axis(find_closest_k_indices, 1, dist_matrix, r+1)

    W = np.zeros((m, m))
    for i in range(top_indices.shape[0]): # row
        for id in top_indices[i]: # id of data point
            if i == id:
                continue
            W[i][id] = 1
            W[id][i] = 1
    # print(W)


    # build D
    diag_vals = np.sum(W, axis=1)
    D = np.zeros((m, m))
    for i in range(diag_vals.shape[0]):
        D[i,i] = diag_vals[i]
    L = D - W

    eigenval, eigenvecs = np.linalg.eig(L)
    eigenvecs_idx = np.argpartition(eigenval, kth=k)[:k]

    return eigenvecs[:, eigenvecs_idx]


def find_closest_k_indices(array, k):
    return np.argpartition(array, kth=k)[:k]