import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def data_clip(data, label, sequence_df):
    for i in range(len(label)):
        min_x, min_y, max_x, max_y, type = label.iloc[i, :5]
        points = []

        sub_data = data[(min_x <= data[1]) & (data[1] <= max_x) & (min_y <= data[2]) & (data[2] <= max_y)]
        for j in range(len(sub_data)):
            channel, x, y = sub_data.iloc[j, :3]
            points.append([x - min_x, y, channel])

        sequence_df = sequence_df.append({'points': points, 'raw_label': type}, ignore_index=True)

    return sequence_df


def label_map(label):
    # label_maps = {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 11: 5, 12: 6, 13: 7}       #8分类（细分正确/细分错误）
    label_maps = {1: 0, 2: 0, 3: 0, 4: 0, 10: 1, 11: 1, 12: 1, 13: 1}  # 2分类（正确/错误）
    # label_maps = {1: 0, 2: 0, 3: 0, 4: 0, 10: 1, 11: 2, 12: 3, 13: 4}

    return label_maps[label]


def str_to_list(s):
    s_list = s.split(',')
    return [float(s) for s in s_list]


def str_to_points(s):
    s = s[1:-1]

    points = []
    temp_s = ''
    in_list = False

    for c in s:
        if c == '[':
            in_list = True
        elif c == ']':
            points.append(str_to_list(temp_s))
            temp_s = ''
            in_list = False
        elif in_list:
            temp_s += c
        else:
            pass

    return points


def process_data(data_save_path):
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

    sequence_df = pd.DataFrame(columns=['points', 'raw_label'])

    for name in names:
        data = pd.read_csv(os.path.join(data_save_path, name + '_arr.txt'), header=None)

        label = pd.read_csv(os.path.join(data_save_path, name + '_label.txt'), header=None)
        label = label.drop_duplicates([0, 1, 2, 3, 4])

        sequence_df = data_clip(data, label, sequence_df)

    sequence_df['length'] = sequence_df['points'].map(lambda x: len(x))
    # sequence_df['new_label'] = sequence_df['raw_label'].map(label_map)

    sequence_df.to_csv(os.path.join(data_save_path, 'data.csv'), index=False)


def read_data(data_save_path):
    data_path = os.path.join(data_save_path, 'data.csv')
    if not os.path.exists(data_path):
        process_data(data_save_path)

    sequence_df = pd.read_csv(data_path)
    sequence_df['points'] = sequence_df['points'].map(str_to_points)

    max_length = sequence_df['length'].max()

    return sequence_df, max_length


def pointwise_pooling_and_generate_image(points):
    '''
    :param points: [N, 3]   x, y, channel
    :param crop_size: crop_height, crop_width
    :return: [crop_height, crop_width, 1]
    '''
    x, y, channel = tf.unstack(points, axis=-1)
    min_x, min_y, max_x, max_y = tf.reduce_min(x), tf.reduce_min(y), tf.reduce_max(x), tf.reduce_max(y)
    width, height = max_x - min_x, max_y - min_y
    crop_height, crop_width = 224, 224

    stride_h = height / crop_height
    stride_w = width / crop_width

    range_h = tf.range(crop_height, dtype=tf.float32)
    range_w = tf.range(crop_width, dtype=tf.float32)

    intervals_h = tf.concat([tf.expand_dims(range_h * stride_h, axis=-1),
                             tf.expand_dims((range_h + 1) * stride_h, axis=-1)], axis=-1)
    intervals_w = tf.concat([tf.expand_dims(range_w * stride_w, axis=-1),
                             tf.expand_dims((range_w + 1) * stride_w, axis=-1)], axis=-1)

    intervals_h_size = tf.shape(intervals_h)[0]
    intervals_w_size = tf.shape(intervals_w)[0]

    intervals_h = tf.tile(tf.expand_dims(intervals_h, axis=1), [1, intervals_w_size, 1])
    intervals_w = tf.tile(tf.expand_dims(intervals_w, axis=0), [intervals_h_size, 1, 1])

    intervals = tf.concat([intervals_h, intervals_w], axis=-1)

    def get_channel(interval):
        assert interval.get_shape().as_list() == [4]

        mask = (interval[0] <= y) & (y <= interval[1]) & (interval[2] <= x) & (x <= interval[3])
        vaild_channel = tf.boolean_mask(channel, mask)

        return tf.cond(tf.size(vaild_channel) > 0,
                       lambda: tf.reduce_mean(vaild_channel),
                       lambda: tf.constant(0, dtype=tf.float32))

    flatten_image = tf.map_fn(get_channel, tf.reshape(intervals, [-1, 4]), dtype=tf.float32)
    image = tf.reshape(flatten_image, [intervals_h_size, intervals_w_size, 1])

    image = tf.image.flip_up_down(image)
    image = tf.squeeze(image, axis=-1)

    return tf.cast(image, dtype=tf.float32)


def points_to_image(points):
    interval_constant = 230.92
    try:
        points = pd.DataFrame(points)
        points[0] = points[0] - points[0].min()
        # points[1] = points[1].map(lambda x: x if x < interval_constant else x - interval_constant)
        points[1] = points[1] - points[1].min()
        points[2] = points[2] + 1
        points[2] = points[2].map(lambda x: x if x <= 9 else x - 9)

        points = tf.constant(points.values, dtype=tf.float32)
        image = pointwise_pooling_and_generate_image(points)
        return tf.Session().run(image)
    except:
        print('error!')
        return None


def points_to_image_and_resize(points):
    '''
    :param points: [N, 3]   x, y
    :return: [h, w, 1]
    '''
    points = pd.DataFrame(points)
    points[0] = points[0] - points[0].min()
    points[1] = points[1] - points[1].min()
    points[2] = points[2] + 1
    points[2] = points[2].map(lambda x: x if x <= 9 else x - 9)

    x = tf.constant(np.round(points[0].values), dtype=tf.int64)
    y = tf.constant(np.round(points[1].values), dtype=tf.int64)
    channel = tf.constant(points[2].values, dtype=tf.int64)

    max_x, max_y = tf.reduce_max(x), tf.reduce_max(y)

    tensor_indices = tf.stack([max_y - y, x], axis=-1)

    sparse_image = tf.sparse.SparseTensor(tensor_indices, values=channel, dense_shape=[max_y + 1, max_x + 1])
    image = tf.sparse.to_dense(sparse_image, default_value=0, validate_indices=False)
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    #image = tf.image.resize_area(image, size=[224, 224])

    return tf.cast(tf.squeeze(image, axis=0), dtype=tf.float32)


def generate_images(data_save_path='./data'):
    images_path = './images_channel'
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    sequence_df, _ = read_data(data_save_path)

    points = sequence_df['points']
    labels = sequence_df['new_label']

    image_no = 1
    for i in tqdm(range(len(points))):
        try:
            label_path = os.path.join(images_path, str(labels[i]))
            if not os.path.exists(label_path):
                os.mkdir(label_path)

            image = pd.DataFrame(points_to_image_and_resize(points[i]))
            image.to_csv(os.path.join(label_path, 'img%d.csv' % image_no), header=False, index=False)
        except:
            print('error to generate img%d' % image_no)
            continue

        image_no += 1


def main():
    generate_images()


if __name__ == '__main__':
    main()
