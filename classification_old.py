import pandas as pd
import tensorflow as tf
import data_preprocess
import models
import os
import math

from models import extract_points_feature
from new_model import extract_images_feature


def generator(sequence_df, batch_size, max_length):
    data_size = len(sequence_df)
    batch = 0
    while (1):
        if batch + batch_size < data_size:
            size = batch_size
            batch_data = sequence_df.iloc[batch:batch + batch_size]

            batch = (batch + batch_size) % data_size
        else:
            size = batch_size - (data_size - 1 - batch)
            batch_data = sequence_df.iloc[batch:data_size - 1]
            batch_data = pd.concat([batch_data, sequence_df.sample(batch_size - size)])

            batch = 0

        xs = batch_data['points']
        xs = tf.keras.preprocessing.sequence.pad_sequences(xs, max_length, padding='post')

        lengths = batch_data['length']

        ys = batch_data['new_label']

        yield (xs, ys, lengths, size)


def train_test(saving, summary, test):
    class_num = 2
    regularizer = 5e-5
    learning_rate = 0.005
    epochs = 20
    train_size = 1600 if test else 2022
    test_size = 422
    batch_size = 200
    train_steps = math.ceil(train_size / batch_size)
    test_steps = math.ceil(test_size / batch_size)
    data_save_path = './data'
    log_path = './logs'
    model_save_path = './model_saving'
    model_name = 'cnn'

    # get dataset and define dataset iters
    sequence_df, max_length = data_preprocess.read_data(data_save_path)
    if test:
        sequence_df = sequence_df.sample(frac=1)

        train_data = sequence_df.iloc[:train_size - 1, :]
        test_data = sequence_df.iloc[train_size:, :]

        train_iter = generator(train_data, batch_size, max_length)
        test_iter = generator(test_data, batch_size, max_length)
    else:
        train_iter = generator(sequence_df, batch_size, max_length)
        test_iter = None

    # define inputs and outputs
    inputs = tf.placeholder(tf.float32, [None, max_length, 3])
    labels = tf.placeholder(tf.int64, [None])
    test_batch_size = tf.placeholder(tf.int64)
    training = tf.placeholder(bool)

    # model = models.CNN(class_num, regularizer)
    # outputs = model.output(inputs, training)
    outputs = extract_points_feature(inputs,class_num,training)

    # variables of train
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(labels, class_num), logits=outputs))
        tf.summary.scalar('mean_cross_entropy_loss', loss)

    with tf.name_scope('accuracy'):
        train_accuracy = tf.metrics.accuracy(labels, tf.argmax(tf.nn.softmax(outputs), axis=1))[1]
        tf.summary.scalar('train_accuracy', train_accuracy)

        test_accuracy = tf.metrics.accuracy(labels[:test_batch_size - 1],
                                            tf.argmax(tf.nn.softmax(outputs[:test_batch_size - 1]), axis=1))[1]
        tf.summary.scalar('test_accuracy', test_accuracy)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    if saving:
        saver = tf.train.Saver()
    else:
        saver = None

    # train and test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if summary:
            writer = tf.summary.FileWriter(log_path, sess.graph)
            merged = tf.summary.merge_all()

        for epoch in range(epochs):
            # train
            for i in range(train_steps):
                xs, ys, lengths, size = next(train_iter)
                feed_dict = {inputs: xs, labels: ys, training: True}

                if summary:
                    _, los, acc, steps, mer = sess.run([train_op, loss, train_accuracy, global_step, merged],
                                                       feed_dict=feed_dict)
                    writer.add_summary(mer, steps)
                else:
                    _, los, acc, steps = sess.run([train_op, loss, train_accuracy, global_step], feed_dict=feed_dict)

                print('epoch %d, step %d: loss is %g, accuracy is %g' % (epoch + 1, steps, los, acc))

                if saving:
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=steps)

            # test
            if test:
                for i in range(test_steps):
                    xs, ys, lengths, size = next(test_iter)
                    feed_dict = {inputs: xs, labels: ys, test_batch_size: size, training: False}

                    acc = sess.run(test_accuracy, feed_dict=feed_dict)
                print('epoch %d: accuracy is %g\n\n' % (epoch + 1, acc))


if __name__ == '__main__':
    train_test(saving=False, summary=False, test=True)
