import os
import pandas as pd
import matplotlib.pyplot as plt

color_dict = {1: 'red', 2: 'orangered', 3: 'yellow', 4: 'green',
              10: 'deepskyblue', 11: 'darkgrey', 12: 'darkviolet', 13: 'teal'}


def draw_box(x1, y1, x2, y2, color):
    plt.plot([x1, x2, x2, x1, x1],
             [y1, y1, y2, y2, y1],
             color=color)


def draw_points(data, min_x, max_x):
    data = data.drop_duplicates([0, 1, 2])
    data = data[(min_x <= data[1]) & (data[1] <= max_x) & (min_x <= data[2]) & (data[2] <= max_x)]

    channels = list(data[0].value_counts().index)

    for i in channels:
        plt.scatter(data[data[0] == i].iloc[:, 1], data[data[0] == i].iloc[:, 2], marker='.')


def draw_labels(label, min_x, max_x):
    label = label.drop_duplicates([0, 1, 2, 3])
    label = label[(min_x <= label[0]) & (label[0] <= max_x) & (min_x <= label[2]) & (label[2] <= max_x)]

    for _, row in label.iterrows():
        draw_box(row[0], row[1], row[2], row[3], color_dict[row[4]])


def draw_picture(data, label, length):
    min_x = data.iloc[:length, 1].min()
    max_x = data.iloc[:length, 1].max()

    plt.figure(figsize=(length / 20, 6.5))

    draw_points(data.iloc[:length, :], min_x, max_x)
    draw_labels(label.iloc[:length, :], min_x, max_x)

    plt.show()


def draw_figures(data_save_path):
    names = []
    for name in os.listdir(data_save_path):
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

    for name in names:
        data = pd.read_csv(os.path.join(data_save_path, name + '_arr.txt'), header=None)
        label = pd.read_csv(os.path.join(data_save_path, name + '_label.txt'), header=None)
        length = 1000

        draw_picture(data, label, length)


def main():
    data_save_path = 'D:/360data/重要数据/桌面/项目/轨道判伤/data'
    draw_figures(data_save_path)


if __name__ == '__main__':
    main()
