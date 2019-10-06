import os
import pandas as pd
import matplotlib.pyplot as plt


def draw_box(x1, y1, x2, y2):
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])


def draw_picture(points, boxes):
    '''
    :param points: [N, 2]
    :param boxes: [M, 4]
    '''
    plt.figure(figsize=(len(points), 3))

    plt.scatter(points.iloc[:, 0], points.iloc[:, 1], marker='.')

    for _, row in boxes.iterrows():
        draw_box(row[0], row[1], row[2], row[3])

    plt.show()


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new_data')
    count = 0

    for name in os.listdir(data_path):
        for part in range(6):
            try:
                points = pd.read_csv(os.path.join(data_path, name, 'part-%d_points.csv' % part),
                                     header=None, engine='python')
                ground_truth = pd.read_csv(os.path.join(data_path, name, 'part-%d_ground_truth.csv' % part),
                                           header=None, engine='python')
            except:
                continue

            draw_picture(points, ground_truth.iloc[:, :4])

        count += 1
        if count > 1:
            break


if __name__ == '__main__':
    main()
