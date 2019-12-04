import os
import tensorflow as tf
from tqdm import tqdm

from models.model import Pointwise_RCNN
from preprocessing.preprocess_data import divide_data, read_data
from models.metric import average_precision


class Trainer:
    def __init__(self):
        self.classes_num = 2
        self.points_num = 1000
        self.max_degree = 1
        self.train_size = 0.8
        self.raw_data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/new_data'
        self.divided_data_path = 'D:/360data/重要数据/桌面/Pointwise-FasterRCNN/divided_data'
        if not os.path.exists(self.divided_data_path):
            divide_data(self.raw_data_path, self.divided_data_path, self.points_num, self.max_degree, self.train_size)

        self.epochs = 20
        self.learning_rate = 0.001
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.model = Pointwise_RCNN()
        # self.saver = tf.train.Saver()

        self.points = tf.placeholder(tf.float32, [self.points_num, 3])
        self.polynomials = tf.placeholder(tf.float32, [self.max_degree + 1, self.points_num, self.points_num])
        self.gt_boxes = tf.placeholder(tf.float32, [None, 4])
        self.gt_labels = tf.placeholder(tf.int32, [None])
        self.max_size = tf.placeholder(tf.float32, [2])  # max_x, max_y

        self.train_data, self.test_data = read_data(self.divided_data_path)

    def train(self):
        print('start training..')

        global_step = tf.Variable(0, trainable=False, name='global_step')
        loss, train_accuracy = self.model.build_network(self.points, self.polynomials, self.gt_boxes, self.gt_labels,
                                                        self.max_size, self.classes_num, training=True)
        train_step = self.optimizer.minimize(loss, global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(self.epochs):
                for i in tqdm(range(len(self.train_data))):
                    ps, gt, poly = self.train_data[i]
                    max_x, max_y = ps[:, 0].max(), ps[:, 1].max()
                    feed_dict = {self.points: ps, self.polynomials: poly, self.gt_boxes: gt[:, :4],
                                 self.gt_labels: gt[:, 4], self.max_size: (max_x, max_y)}

                    _, loss_val, train_acc_val = sess.run([train_step, loss, train_accuracy], feed_dict=feed_dict)

                    print('epoch %d, loss is %.2f, train acc is %.2f' % (epoch, loss_val, train_acc_val))


def main():
    model = Trainer()
    model.train()


if __name__ == '__main__':
    main()
