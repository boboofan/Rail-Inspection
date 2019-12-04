import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import KNeighborsRegressor


def calculate_distance(point1, point2, max_distance, sigma):
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]

    squared_distance = np.square(x1 - x2) + np.square(y1 - y2)
    if np.sqrt(squared_distance) > max_distance:
        return 0
    else:
        return np.exp(-squared_distance / np.square(sigma))  # Gaussian kernel


def generate_adjacency(points, max_distance, sigma):
    points_num = np.shape(points)[0]
    adjacency = np.eye(points_num)

    for i in range(points_num):
        for j in range(i + 1, points_num):
            distance = calculate_distance(points[i], points[j], max_distance, sigma)
            adjacency[i][j], adjacency[j][i] = distance, distance

    return adjacency


def normalized_laplacian(A):
    D_power = np.diag(np.power(np.sum(A, axis=1), -0.5))
    return np.eye(np.shape(A)[0]) - np.matmul(np.matmul(D_power, A), D_power)


def rescaled_laplacian(L):
    try:
        largest_eigenvalue = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
    except:
        largest_eigenvalue = 2

    return (2.0 / largest_eigenvalue) * L - np.eye(np.shape(L)[0])


def chebyshev_polynomial(X, K):
    T_k_list = []
    T_k_list.append(np.eye(np.shape(X)[0]))
    T_k_list.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two):
        return np.matmul(2 * X, T_k_minus_one) - T_k_minus_two

    for k in range(2, K + 1):
        T_k_list.append(chebyshev_recurrence(T_k_list[-1], T_k_list[-2]))

    return T_k_list


def get_chebyshev_polynomials(points, max_degree, max_distance=91, sigma=50):
    print('calculate polynomials..')
    adjacency = generate_adjacency(points, max_distance, sigma)
    L = normalized_laplacian(adjacency)
    rescaled_L = rescaled_laplacian(L)
    polynomials = chebyshev_polynomial(rescaled_L, max_degree)

    return np.array(polynomials,dtype=np.float32)


def kNN_sampling(points, sampling_num, n_neighbors=2):
    points_num = np.shape(points)[0]
    if points_num == sampling_num:
        return points
    elif points_num > sampling_num:
        choices = np.arange(points_num)
        choices = np.random.choice(choices, sampling_num, replace=False)
        choices.sort()
        return points[choices]
    else:
        KNR = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        KNR.fit(points[:, 0].reshape([-1, 1]), points[:, [1, 2]])

        sampling_num -= points_num
        max_x, min_x = points[:, 0].max(), points[:, 0].min()
        delta = (max_x - min_x) / (sampling_num + 2)
        sampling_x = np.linspace(min_x + delta, max_x - delta, sampling_num).reshape([-1, 1])
        sampling_yc = KNR.predict(sampling_x)
        sampling_points = np.concatenate([sampling_x, sampling_yc], axis=-1)

        return np.concatenate([points, sampling_points], axis=0)


def crop_and_sample_points(points, boxes, points_num):
    '''
    :param points: [P, 3] x, y, c
    :param boxes: [B, 4] min_x, min_y, max_x, max_y
    :return: [B, points_num, 3]
    '''
    if np.shape(points)[0] == 0 or np.shape(boxes)[0] == 0:
        return np.zeros([np.shape(boxes)[0], points_num, 3], dtype=np.float32)

    points = points[np.argsort(points[:, 0])]
    boxes = boxes[np.argsort(boxes[:, 0])]

    batch_points = []
    i = 0
    for box in boxes:
        min_x, min_y, max_x, max_y = box
        while i < np.shape(points)[0] and points[i, 0] < min_x:
            i += 1

        cropped_points = []
        j = i
        while j < np.shape(points)[0] and points[j, 0] <= max_x:
            if min_y <= points[j, 1] and points[j, 1] <= max_y:
                cropped_points.append(points[j])

        if len(cropped_points) > 0:
            cropped_points = kNN_sampling(np.array(cropped_points), points_num)
        else:
            cropped_points = np.zeros([points_num, 3], dtype=np.float32)
        batch_points.append(cropped_points)

    return np.array(batch_points)
