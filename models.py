import tensorflow as tf


def conv1d(inputs, filters=None, kernel_size=3, strides=1, padding='same', activation=None):
    if not filters:
        filters = inputs.get_shape().as_list()[-1]
    return tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def max_pooling1d(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling1d(inputs, pool_size, strides, padding)


def avg_pooling1d(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.average_pooling1d(inputs, pool_size, strides, padding)


def res_block(inputs, output_dim, strides, training):
    input_dim = inputs.get_shape().as_list()[-1]

    if input_dim == output_dim:
        if strides == 2:
            shortcut = max_pooling1d(inputs)
        else:
            shortcut = inputs
    else:
        shortcut = conv1d(inputs, output_dim, 1, strides)

    conv1 = conv1d(inputs, output_dim, strides=strides)
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    conv2 = conv1d(relu1, output_dim)
    bn2 = tf.layers.batch_normalization(conv2, training=training)
    relu2 = tf.nn.relu(bn2)

    return tf.add(shortcut, relu2)


def extract_points_feature(points, classes_num, training):
    '''
    :param points: [batch, N, 3]
    '''

    res1 = res_block(points, 16, 1, training)
    pool1 = avg_pooling1d(res1, strides=1)

    res2 = res_block(pool1, 32, 1, training)
    pool2 = avg_pooling1d(res2, strides=1)

    res3 = res_block(pool2, 64, 1, training)
    pool3 = avg_pooling1d(res3, strides=1)

    flatten = tf.layers.flatten(pool3)
    fc = tf.layers.dense(flatten, units=1000, activation=tf.nn.relu)
    return tf.layers.dense(fc, units=classes_num)

# class CNN:
#     def __init__(self, class_num, regularizer):
#         self.class_num = class_num
#         self.regularizer = regularizer
#
#     def conv1d(self, inputs, filters, kernel_size, padding='same', activation=tf.nn.relu):
#         return tf.layers.conv1d(inputs,
#                                 filters=filters,
#                                 kernel_size=kernel_size,
#                                 padding=padding,
#                                 activation=activation,
#                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#
#     def stack_block(self, inputs, output_dim1, output_dim2, training):
#         bn = tf.layers.batch_normalization(inputs, training=training)
#         conv1 = self.conv1d(bn, output_dim1, 1)
#         conv2 = self.conv1d(conv1, output_dim1, 3)
#         conv3 = self.conv1d(conv2, output_dim2, 1)
#         return conv3
#
#     def output(self, inputs, training):
#         stack1 = self.stack_block(inputs, 16, 32, training)
#
#         stack2 = self.stack_block(stack1, 32, 64, training)
#
#         flatten = tf.layers.flatten(stack2)
#
#         fc = tf.layers.dense(flatten, 1000, activation=tf.nn.relu,
#                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#         fc = tf.layers.dropout(fc, 0.5)
#
#         return tf.layers.dense(fc, self.class_num, activation=None)
#
#     def output(self, inputs, sequence_lengths, max_length):
#         conv1 = self.conv1d(inputs, 8, 3)
#
#         max_pool1 = tf.layers.max_pooling1d(conv1, 2, 1, 'same')
#
#         conv2 = self.conv1d(max_pool1, 16, 3)
#
#         max_pool2 = tf.layers.max_pooling1d(conv2, 2, 1, 'same')
#
#         cell_fw = tf.nn.rnn_cell.MultiRNNCell(
#             [tf.nn.rnn_cell.LSTMCell(self.cell_dim, state_is_tuple=True) for _ in range(self.cell_num)],
#             state_is_tuple=True)
#         cell_bw = tf.nn.rnn_cell.MultiRNNCell(
#             [tf.nn.rnn_cell.LSTMCell(self.cell_dim, state_is_tuple=True) for _ in range(self.cell_num)],
#             state_is_tuple=True)
#
#         (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, max_pool2,
#                                                                     sequence_length=sequence_lengths, dtype=tf.float32)
#
#         mask = tf.sequence_mask(sequence_lengths, max_length)
#         mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, output_fw.shape[2]])
#
#         mask_output_fw = tf.where(mask, output_fw, tf.zeros_like(output_fw))
#         mask_output_bw = tf.where(mask, output_bw, tf.zeros_like(output_bw))
#
#         outputs = tf.reduce_sum(tf.concat([mask_output_fw, mask_output_bw], axis=1), axis=1)
#
#         fc = tf.layers.dense(outputs, 100, activation=tf.nn.relu,
#                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#
#         return tf.layers.dense(fc, self.class_num)
