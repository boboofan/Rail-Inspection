import pandas as pd
import os


def label_map(label):
    # label_maps = {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 11: 5, 12: 6, 13: 7}       #8分类（细分正确/细分错误）
    label_maps = {0: 0, 1: 0, 2: 0, 3: 0, 10: 1, 11: 1, 12: 1, 13: 1}  # 2分类（正确/错误）
    # label_maps = {1: 0, 2: 0, 3: 0, 4: 0, 10: 1, 11: 2, 12: 3, 13: 4}

    return label_maps[label]


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

    return names


def convert_data_format(old_data_path, new_data_path, names, invalid_values, distance_constant):
    for name in names:
        # points
        points = pd.read_csv(os.path.join(old_data_path, name + '_arr.txt'), header=None, engine='python')
        points = points.iloc[:, :3]  # channel, x, y
        points = points[~points.iloc[:, 2].isin(invalid_values)]

        points_above = points[points[2] >= distance_constant]
        points_above[2] = points_above[2].map(lambda y: y - distance_constant)

        points_underneath = points[points[2] < distance_constant]

        # ground_truth
        ground_truth = pd.read_csv(os.path.join(old_data_path, name + '_label.txt'), header=None, engine='python')
        ground_truth = ground_truth.iloc[:, :5]  # min_x, min_y, max_x, max_y, label
        ground_truth[4] = ground_truth[4].map(label_map)

        ground_truth_above = ground_truth[ground_truth[1] >= distance_constant]
        ground_truth_above[1] = ground_truth_above[1].map(lambda y: y - distance_constant)
        ground_truth_above[3] = ground_truth_above[3].map(lambda y: y - distance_constant)

        ground_truth_underneath = ground_truth[ground_truth[3] <= distance_constant]

        # save
        path = os.path.join(new_data_path, name)
        if not os.path.exists(path):
            os.mkdir(path)

        points_above.to_csv(os.path.join(path, 'points_above.csv'), header=False, index=False)
        points_underneath.to_csv(os.path.join(path, 'points_underneath.csv'), header=False, index=False)
        ground_truth_above.to_csv(os.path.join(path, 'ground_truth_above.csv'), header=False, index=False)
        ground_truth_underneath.to_csv(os.path.join(path, 'ground_truth_underneath.csv'), header=False, index=False)

        # for i in range(len(intervals)):
        #     interval = intervals[i]
        #
        #     sub_points = points[(interval[0] <= points.iloc[:, 1]) &
        #                         (points.iloc[:, 1] <= interval[1]) &
        #                         (~points.iloc[:, 1].isin(invalid_values))]
        #     sub_points = sub_points.sort_values(by=1)
        #
        #     sub_ground_truth = ground_truth[(interval[0] <= ground_truth.iloc[:, 1]) &
        #                                     (ground_truth.iloc[:, 3] <= interval[1])]
        #     sub_ground_truth = sub_ground_truth.sort_values(by=0)
        #
        #     path = os.path.join(new_data_path, name)
        #     if not os.path.exists(path):
        #         os.mkdir(path)
        #
        #     sub_points.to_csv(os.path.join(path, 'part-' + str(i) + '_points.csv'), index=False, header=False)
        #     sub_ground_truth.to_csv(os.path.join(path, 'part-' + str(i) + '_ground_truth.csv'),
        #                             index=False, header=False)


def read_dataset(data_path='./new_data', slice_distance=10000, buffer_size=70):
    for name in os.listdir(data_path):
        try:
            points_above = pd.read_csv(os.path.join(data_path, name, 'points_above.csv'),
                                       header=None, engine='python')

            points_underneath = pd.read_csv(os.path.join(data_path, name, 'points_underneath.csv'),
                                            header=None, engine='python')

            ground_truth_above = pd.read_csv(os.path.join(data_path, name, 'ground_truth_above.csv'),
                                             header=None, engine='python')

            ground_truth_underneath = pd.read_csv(os.path.join(data_path, name, 'ground_truth_underneath.csv'),
                                                  header=None, engine='python')

            data_list = [[points_above, ground_truth_above], [points_underneath, ground_truth_underneath]]
        except:
            continue

        for points, ground_truth in data_list:
            # points: channel, x, y
            # ground_truth: min_x, min_y, max_x, max_y, label

            start, end = 0, slice_distance

            while start <= points.iloc[-1, 1]:
                slice_points = points[(start <= points.iloc[:, 1]) & (points.iloc[:, 1] < end)]
                slice_ground_truth = ground_truth[(start <= ground_truth.iloc[:, 0]) & (ground_truth.iloc[:, 2] < end)]

                min_x, min_y = slice_points.iloc[:, 1].min(), slice_points.iloc[:, 2].min()
                max_x, max_y = slice_points.iloc[:, 1].max(), slice_points.iloc[:, 2].max()

                slice_points.iloc[:, 1] = slice_points.iloc[:, 1].map(lambda x: x - min_x)
                slice_points.iloc[:, 2] = slice_points.iloc[:, 2].map(lambda y: y - min_y)

                slice_ground_truth.iloc[:, [0, 2]] = slice_ground_truth.iloc[:, [0, 2]].applymap(lambda x: x - min_x)
                slice_ground_truth.iloc[:, [1, 3]] = slice_ground_truth.iloc[:, [1, 3]].applymap(lambda y: y - min_y)
                slice_ground_truth.iloc[:, 4] = slice_ground_truth.iloc[:, 4].map(label_map)

                slice_gt_boxes, slice_gt_labels = slice_ground_truth.iloc[:, :4], slice_ground_truth.iloc[:, 4]
                slice_gt_labels = slice_gt_labels + 1

                orgin_point = [min_x, min_y]
                max_size = [max_x - min_x, max_y - min_y]

                start, end = start + slice_distance - buffer_size, end + slice_distance - buffer_size

                yield slice_points, slice_gt_boxes, slice_gt_labels, orgin_point, max_size


def main():
    old_data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/old_data'
    assert os.path.exists(old_data_path)

    new_data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/new_data'
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)

    # 0.05740115894332741, 7.886573546679733
    # 175.6800, 406.6000
    # {2: [254.879, 291.4796], 1: [296.0156, 337.6863], 13: [257.0972, 288.7714], 3: [241.8476, 312.0387], 12: [297.0251, 358.5923], 10: [241.8476, 299.7553], 11: [399.0269, 415.1806], 4: [314.0033, 370.9769]}
    # {2: [22.502, 51.5237], 1: [nan, nan], 13: [23.6112, 51.8767], 3: [10.0228, 77.1272], 12: [66.0678, 130.1056], 10: [10.0228, 69.2671], 11: [170.446, 185.1806], 4: [nan, nan]}
    intervals = [[0, 80], [60, 135], [170, 190], [240, 315], [290, 375], [395, 420]]
    invalid_values = [175.68, 406.60]
    distance_constant = 230.92

    names = analyze_data_name(old_data_path)
    convert_data_format(old_data_path, new_data_path, names, invalid_values, distance_constant)


if __name__ == '__main__':
    main()
