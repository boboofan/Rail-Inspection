import numpy as np
import pandas as pd
import os


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


def process_data(data_path, new_data_path, names, intervals, invalid_values):
    for name in names:
        points = pd.read_csv(os.path.join(data_path, name + '_arr.txt'), header=None)
        points = points.iloc[:, 1:3]  # x, y

        ground_truth = pd.read_csv(os.path.join(data_path, name + '_label.txt'), header=None)
        ground_truth = ground_truth.iloc[:, :5]  # min_x, min_y, max_x, max_y, label

        for i in range(len(intervals)):
            interval = intervals[i]

            sub_points = points[(interval[0] <= points.iloc[:, 1]) &
                                (points.iloc[:, 1] <= interval[1]) &
                                (~points.iloc[:, 1].isin(invalid_values))]
            sub_points = sub_points.sort_values(by=1)

            sub_ground_truth = ground_truth[(interval[0] <= ground_truth.iloc[:, 1]) &
                                            (ground_truth.iloc[:, 3] <= interval[1])]
            sub_ground_truth = sub_ground_truth.sort_values(by=0)

            path = os.path.join(new_data_path, name)
            if not os.path.exists(path):
                os.mkdir(path)

            sub_points.to_csv(os.path.join(path, 'part-' + str(i) + '_points.csv'), index=False, header=False)
            sub_ground_truth.to_csv(os.path.join(path, 'part-' + str(i) + '_ground_truth.csv'),
                                    index=False, header=False)


def main():
    data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/old_data'
    assert os.path.exists(data_path)

    new_data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/new_data'
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)

    # 175.6800,406.6000
    # {2: [254.879, 291.4796], 1: [296.0156, 337.6863], 13: [257.0972, 288.7714], 3: [241.8476, 312.0387], 12: [297.0251, 358.5923], 10: [241.8476, 299.7553], 11: [399.0269, 415.1806], 4: [314.0033, 370.9769]}
    # {2: [22.502, 51.5237], 1: [nan, nan], 13: [23.6112, 51.8767], 3: [10.0228, 77.1272], 12: [66.0678, 130.1056], 10: [10.0228, 69.2671], 11: [170.446, 185.1806], 4: [nan, nan]}
    intervals = [[0, 80], [60, 135], [170, 190], [240, 315], [290, 375], [395, 420]]
    invalid_values = [175.6800, 406.6000]

    names = analyze_data_name(data_path)
    process_data(data_path, new_data_path, names, intervals, invalid_values)


if __name__ == '__main__':
    main()
