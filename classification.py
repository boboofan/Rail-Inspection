import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from new_model import extract_images_feature, cnn


def read_data(data_path='./images_channel', train_size=0.8):
    images_path, labels = [], []

    labels_list = os.listdir(data_path)
    for label in labels_list:
        path = os.path.join(data_path, label)
        img_names = os.listdir(path)
        img_paths = list(map(lambda name: os.path.join(path, name), img_names))
        img_labels = [int(label)] * len(img_paths)

        images_path += img_paths
        labels += img_labels

    return train_test_split(images_path, labels, train_size=train_size)


def data_iter(X, Y, batch_size):
    batch = 0
    while batch < len(X):
        if batch + batch_size < len(X):
            xs, ys = X[batch:batch + batch_size], Y[batch: batch + batch_size]
        else:
            xs, ys = X[batch:], Y[batch:]

        xs = [pd.read_csv(path, header=None, engine='python').values for path in xs]
        xs = np.array(xs)

        batch += batch_size

        yield xs, ys


def main():
    epochs = 10
    train_batch_size = 100
    test_batch_size = 100
    learning_rate = 0.005

    train_x, test_x, train_y, test_y = read_data()

    images = tf.placeholder(tf.float32, [None, 224, 224])
    labels = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool)
    logits = cnn(tf.expand_dims(images, axis=-1), training)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    train_accuracy = tf.metrics.accuracy(labels=labels,
                                         predictions=tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1,
                                                               output_type=tf.int32))[1]
    test_accuracy = tf.equal(labels, tf.argmax(logits, axis=-1, output_type=tf.int32))
    test_accuracy = tf.reduce_mean(tf.cast(test_accuracy, tf.float32))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(epochs):
            train_iter = data_iter(train_x, train_y, train_batch_size)
            test_iter = data_iter(test_x, test_y, test_batch_size)

            print('\n\ntraining:')
            for xs, ys in train_iter:
                feed_dict = {images: xs, labels: ys, training: True}
                _, loss_val, train_acc_val = sess.run([train_step, loss, train_accuracy], feed_dict=feed_dict)

                print('epoch %d, loss is %.2f, accuracy is %.2f' % (epoch, loss_val, train_acc_val))

            print('\ntesting:')
            acc_list = []
            for xs, ys in test_iter:
                feed_dict = {images: xs, labels: ys, training: False}
                acc_list.append(sess.run(test_accuracy, feed_dict=feed_dict))

            print('accuracy is %.2f' % (sum(acc_list) / len(acc_list)))


if __name__ == '__main__':
    main()
