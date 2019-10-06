import tensorflow as tf
import os
from easydict import EasyDict

from models.model import Faster_RCNN
from preprocessing.dataset import read_dataset
from models.metric import average_precision
from preprocessing.draw_figure_new import draw_picture


def get_config():
    cfg = EasyDict()

    cfg.classes_num = 2
    cfg.epochs = 10
    cfg.learning_rate = 0.001
    cfg.log_path = './logs'
    cfg.model_saving_path = './model_saving'
    cfg.model_name = 'Pointwise-FasterRCNN'

    return cfg


def train(cfg, saving):
    points = tf.placeholder(tf.float32, [None, 3])
    gt_boxes = tf.placeholder(tf.float32, [None, 4])
    gt_labels = tf.placeholder(tf.int32, [None])
    max_size = tf.placeholder(tf.float32, [2])

    model = Faster_RCNN(cfg.classes_num)
    loss = model.get_loss(points, gt_boxes, gt_labels, max_size)
    pred_boxes, pred_labels, pred_scores = model.get_outputs(points, gt_boxes, gt_labels, max_size)

    global_steps = tf.Variable(0, trainable=False)

    train_step = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss, global_step=global_steps)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(cfg.epochs):
            iter = read_dataset()
            for input_points, input_gt_boxes, input_gt_labels, input_orgin_point, input_max_size in iter:
                if input_points.shape[0] == 0 or input_gt_boxes.shape[0] == 0:
                    continue

                feed_dict = {points: input_points, gt_boxes: input_gt_boxes,
                             gt_labels: input_gt_labels, max_size: input_max_size}
                _, loss_val, steps = sess.run([train_step, loss, global_steps], feed_dict=feed_dict)
                print('epoch %d: loss is %.2f' % (epoch + 1, loss_val))
                if saving and steps % 100 == 0:
                    saver.save(sess, os.path.join(cfg.model_saving_path, cfg.model_name), global_step=steps)


def main():
    train(get_config(), False)


if __name__ == '__main__':
    main()
