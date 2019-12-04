import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models.model import *
from preprocessing.preprocess_data import divide_data, read_data
from models.metric import average_precision


def date_iter(X, Y, batch_size):
    batch = 0
    while 1:
        if batch + batch_size < len(X):
            xs, ys = X[batch:batch + batch_size], Y[batch:batch + batch_size]
        else:
            xs, ys = X[batch:], Y[batch:]

        batch = batch + batch_size if batch + batch_size < len(X) else 0

        yield xs, ys


def mnist():
    path = 'D:/360data/重要数据/桌面/轨道探伤/tf-keras/mnist_data/mnist.npz'
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data(path)
    train_x = np.reshape(train_x, [-1, 28, 28, 1]).astype(np.float32)
    test_x = np.reshape(test_x, [-1, 28, 28, 1]).astype(np.float32)
    train_y, test_y = train_y.reshape([-1]).astype(np.int32), test_y.reshape([-1]).astype(np.int32)
    train_iter, test_iter = date_iter(train_x, train_y, 100), date_iter(test_x, test_y, 100)

    return train_iter, test_iter


def train():
    classes_num = 2
    points_num = 1000
    epochs = 20
    learning_rate = 0.001

    points = tf.placeholder(tf.float32, [1000, 3])
    polynomials = tf.placeholder(tf.float32, [2, points_num, points_num])
    gt_boxes = tf.placeholder(tf.float32, [None, 4])
    gt_labels = tf.placeholder(tf.int32, [None])
    max_size = tf.placeholder(tf.float32, [None])

    divided_data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/divided_data'
    train_data, test_data = read_data(divided_data_path)
    model = Pointwise_RCNN()

    anchors = model.generate_anchors(points)
    norm_points = model.normalization(points, batch=False)
    pred_offsets, pred_scores = model.rpn_head(norm_points, polynomials, model.anchors_num)
    pred_boxes = offsets_to_boxes(pred_offsets, anchors)
    proposal_boxes, proposal_scores = model.generate_rpn_proposals(pred_boxes, pred_scores, max_size, True)
    reg_loss, cls_loss = model.rpn_losses(anchors, gt_boxes, pred_offsets, pred_scores)

    # output=reg_loss
    loss = reg_loss + cls_loss
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(1):
            for i in tqdm(range(len(train_data))):
                ps, gt, poly = train_data[i]
                max_x, max_y = ps[:, 0].max(), ps[:, 1].max()
                feed_dict = {points: ps, polynomials: poly, gt_boxes: gt[:, :4],
                             gt_labels: gt[:, 4], max_size: (max_x, max_y)}
                print(sess.run([proposal_boxes, proposal_scores], feed_dict=feed_dict))
                #_, l1, l2 = sess.run([train_step, proposal_boxes, proposal_scores], feed_dict=feed_dict)

                #print('epoch %d, loss is %.2f, %.2f' % (epoch, l1, l2))


def main():
    train()


if __name__ == '__main__':
    main()
