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


def statistic(data_path, names):
    min_x, max_x, min_y, max_y = 10000000, 0, 10000000, 0
    min_box_x, min_box_y, max_box_x, max_box_y = 10000000, 10000000, 0, 0,
    min_w, min_h, max_w, max_h = 10000000, 10000000, 0, 0
    min_r, max_r = 1000, 0
    labels = {}

    ws, hs = [], []

    for name in names:
        points = pd.read_csv(os.path.join(data_path, name + '_arr.txt'), header=None)
        points = points.iloc[:, 1:3]  # x, y

        if points.iloc[:, 0].min() < min_x:
            min_x = points.iloc[:, 0].min()
        if points.iloc[:, 0].max() > max_x:
            max_x = points.iloc[:, 0].max()
        if points.iloc[:, 1].min() < min_y:
            min_y = points.iloc[:, 1].min()
        if points.iloc[:, 1].max() > max_y:
            max_y = points.iloc[:, 1].max()

        ground_truth = pd.read_csv(os.path.join(data_path, name + '_label.txt'), header=None)
        ground_truth = ground_truth.iloc[:, :5]  # min_x, min_y, max_x, max_y, label

        labels_series = ground_truth.iloc[:, 4].value_counts()
        for k, v in dict(labels_series).items():
            if k not in labels:
                labels[k] = v
            else:
                labels[k] += v

        if ground_truth.iloc[:, 0].min() < min_box_x:
            min_box_x = ground_truth.iloc[:, 0].min()
        if ground_truth.iloc[:, 1].min() < min_box_y:
            min_box_y = ground_truth.iloc[:, 1].min()
        if ground_truth.iloc[:, 2].max() > max_box_x:
            max_box_x = ground_truth.iloc[:, 2].max()
        if ground_truth.iloc[:, 3].max() > max_box_y:
            max_box_y = ground_truth.iloc[:, 3].max()

        w = ground_truth.iloc[:, 2] - ground_truth.iloc[:, 0]
        ws.append(w)
        if w.min() < min_w:
            min_w = w.min()
        if w.max() > max_w:
            max_w = w.max()

        h = ground_truth.iloc[:, 3] - ground_truth.iloc[:, 1]
        hs.append(h)
        if h.max() > max_h:
            max_h = h.max()
        if h.min() < min_h:
            min_h = h.min()

        ratio = h / w
        if ratio.min() < min_r:
            min_r = ratio.min()
        if ratio.max() > max_r:
            max_r = ratio.max()

    ws = pd.concat(ws, axis=0)
    hs = pd.concat(hs, axis=0)
    rs = (ws / hs)
    #.map(lambda x: round(x))
    hs = hs.map(lambda x: round(x))
    ws = ws.map(lambda x: round(x))

    # print(ws.value_counts().sort_index())
    # print(hs.value_counts().sort_index())
    print(rs.value_counts().sort_index().index.to_list())

    print(min_x, max_x, min_y, max_y)
    print(min_box_x, min_box_y, max_box_x, max_box_y)
    print(min_w, min_h, max_w, max_h)
    print(min_r, max_r)
    print(sorted(labels.items()))


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'old_data')

    assert os.path.exists(data_path)

    names = analyze_data_name(data_path)
    statistic(data_path, names)


if __name__ == '__main__':
    main()
