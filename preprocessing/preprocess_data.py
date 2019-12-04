import os
import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm

from models.utils import kNN_sampling, get_chebyshev_polynomials


def preprocess_raw_data(raw_data_path='D:/360data/重要数据/桌面/Pointwise-FasterRCNN/old_data',
                        new_data_path='D:/360data/重要数据/桌面/Pointwise-FasterRCNN/new_data'):
    print('preprocessing data..')

    def analyze_data_name(data_path):
        names = []
        for name in os.listdir(data_path):
            if name[-4:] == '.txt':
                name = name[:-4]
                if name[-4:] == '_arr':
                    name = name[:-4]
                elif name[-6:] == '_label':
                    name = name[:-6]
                else:
                    continue

                if name not in names:
                    names.append(name)

        return tuple(names)

    names = analyze_data_name(raw_data_path)
    for name in tqdm(names):
        points = pd.read_csv(os.path.join(raw_data_path, name + '_arr.txt'), header=None, engine='python')
        points = points.iloc[:, :3].values  # channel, x, y
        points = points[:, [1, 2, 0]]  # x, y, channel

        y = points[:, 1]

        special_values = []
        special_values.append(mode(y[y <= 200])[0][0])
        special_values.append(mode(y[y > 200])[0][0])

        mask = np.all([points[:, 1] != special_values[0], points[:, 1] != special_values[1]], axis=0)
        points = points[mask, :]

        low_points = points[points[:, 1] < 200]
        high_points = points[points[:, 1] >= 200]
        high_points[:, 0] += low_points[:, 0].max()
        high_points[:, 1] -= (special_values[1] - special_values[0])
        points = np.concatenate([low_points, high_points], axis=0)

        points = points[np.argsort(points[:, 0])]

        ground_truth = pd.read_csv(os.path.join(raw_data_path, name + '_label.txt'), header=None, engine='python')
        ground_truth = ground_truth.iloc[:, :5].values  # min_x, min_y, max_x, max_y, label

        low_ground_truth = ground_truth[ground_truth[:, 1] < 200]
        high_ground_truth = ground_truth[ground_truth[:, 1] >= 200]
        high_ground_truth[:, [0, 2]] += low_points[:, 0].max()
        high_ground_truth[:, [1, 3]] -= (special_values[1] - special_values[0])
        ground_truth = np.concatenate([low_ground_truth, high_ground_truth], axis=0)

        ground_truth = ground_truth[np.argsort(ground_truth[:, 0])]

        save_path = os.path.join(new_data_path, name)
        assert not os.path.exists(save_path)

        os.makedirs(save_path)
        np.save(os.path.join(save_path, 'points.npy'), points)
        np.save(os.path.join(save_path, 'ground_truth.npy'), ground_truth)


def divide_data(data_path='D:/360data/重要数据/桌面/Pointwise-FasterRCNN/new_data',
                divided_data_path='D:/360data/重要数据/桌面/Pointwise-FasterRCNN/divided_data',
                points_num=1000, max_degree=1, train_size=0.8, random_seed=None):
    print('dividing data..')

    points_list, gt_list, polynomials_list = [], [], []
    names = os.listdir(data_path)

    for name in tqdm(names):
        points = np.load(os.path.join(data_path, name, 'points.npy'))  # x, y, channel
        ground_truth = np.load(os.path.join(data_path, name, 'ground_truth.npy'))  # min_x, min_y, max_x, max_y, label

        segments = len(points) // points_num
        if len(points) % points_num >= (points_num / 3):
            segments += 1
        points = kNN_sampling(points, segments * points_num)

        i = 0
        for _ in range(segments):
            points_seg = points[i:i + points_num]
            min_x, max_x = points_seg[:, 0].min(), points_seg[:, 0].max()
            gt_seg = ground_truth[(min_x <= ground_truth[:, 0]) & (ground_truth[:, 2] <= max_x)]

            points_seg[:, 0] -= min_x
            gt_seg[:, [0, 2]] -= min_x

            points_list.append(points_seg)
            gt_list.append(gt_seg)
            polynomials_list.append(get_chebyshev_polynomials(points_seg, max_degree))

            i += points_num

    choices = np.arange(len(points_list))
    if random_seed:
        np.random.seed(random_seed)
    train_index = np.random.choice(choices, int(len(points_list) * train_size), replace=False)
    test_index = np.delete(choices, train_index)

    train_data = [[points_list[i], gt_list[i], polynomials_list[i]] for i in train_index]
    test_data = [[points_list[i], gt_list[i], polynomials_list[i]] for i in test_index]

    def save_data(root_path, data):
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        for i in range(len(data)):
            path = os.path.join(root_path, 'data%d' % i)
            if not os.path.exists(path):
                os.makedirs(path)

            np.save(os.path.join(path, 'points.npy'), train_data[i][0])
            np.save(os.path.join(path, 'ground_truth.npy'), train_data[i][1])
            np.save(os.path.join(path, 'polynomials.npy'), train_data[i][2])

    if not os.path.exists(divided_data_path):
        os.makedirs(divided_data_path)

    train_path, test_path = os.path.join(divided_data_path, 'train'), os.path.join(divided_data_path, 'test')
    save_data(train_path, train_data), save_data(test_path, test_data)


def read_data(data_path='D:/360data/重要数据/桌面/Pointwise-FasterRCNN/divided_data'):
    print('reading data..')

    def read_file(root_path, names):
        data_list = []
        for name in tqdm(names):
            path = os.path.join(root_path, name)
            a = np.load(os.path.join(path, 'points.npy'))
            b = np.load(os.path.join(path, 'ground_truth.npy'))
            c = np.load(os.path.join(path, 'polynomials.npy'))
            data_list.append([a, b, c])
        return data_list

    train_path, test_path = os.path.join(data_path, 'train'), os.path.join(data_path, 'test')
    train_data_names, test_data_names = os.listdir(train_path), os.listdir(test_path)
    train_data, test_data = read_file(train_path, train_data_names), read_file(test_path, test_data_names)

    return train_data, test_data


def main():
    # preprocess_raw_data()
    divide_data()


if __name__ == '__main__':
    main()
