import numpy as np


def count_dist_matrix(xs, m, N):
    dist_matrix = np.zeros((N - m + 1, N - m + 1))
    for i in range(N - m + 1):
        for j in range(i, N - m + 1):
            dist_matrix[i, j] = dist_matrix[j, i] = \
                max([abs(xs[i][k] - xs[j][k]) for k in range(m)])
    return dist_matrix


def count_entropy(series: np.ndarray):
    r = 1
    N = series.shape[0]

    def _phi(m):
        xs = [series[i:i+m] for i in range(N - m + 1)]
        dist_matrix = count_dist_matrix(xs, m, N)

        C_m = []
        for i in range(N - m + 1):
            n_less = len(list(filter(lambda x: x <= r, dist_matrix[i, :])))
            if n_less == 0:
                continue
            C_m.append(np.log(n_less / (N - m + 1)))

        return sum(C_m) / (N - m + 1)

    return _phi(2) - _phi(3)
