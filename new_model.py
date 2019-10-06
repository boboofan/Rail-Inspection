import tensorflow as tf


def conv2d(inputs, filters=None, kernel_size=3, strides=1, padding='same', activation=None):
    if not filters:
        filters = inputs.get_shape().as_list()[-1]
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-5))


def max_pooling2d(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def res_block(inputs, output_dim, strides, training):
    input_dim = inputs.get_shape().as_list()[-1]

    if input_dim == output_dim:
        if strides == 2:
            shortcut = max_pooling2d(inputs)
        else:
            shortcut = inputs
    else:
        shortcut = conv2d(inputs, output_dim, 1, strides)

    conv1 = conv2d(inputs, output_dim, strides=strides)
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    conv2 = conv2d(relu1, output_dim)
    bn2 = tf.layers.batch_normalization(conv2, training=training)
    relu2 = tf.nn.relu(bn2)

    return tf.add(shortcut, relu2)


def cnn(images, training):
    bn = tf.layers.batch_normalization(images, training=training)

    x = bn
    filters = [16, 32, 64, 128, 256]
    for i in range(5):
        x = res_block(x, filters[i], 1, training)
        x = max_pooling2d(x)
    x = tf.layers.average_pooling2d(x, pool_size=7, strides=1)

    flatten = tf.layers.flatten(x)
    fc = tf.layers.dense(flatten, 200, activation=tf.nn.relu)
    return tf.layers.dense(fc, 2)


def extract_images_feature(images, training):
    conv1 = conv2d(images, 64, 7, 2)
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    pool1 = tf.layers.max_pooling2d(relu1, 3, 2, 'same')
    res2_1 = res_block(pool1, 64, 1, training)
    res2_2 = res_block(res2_1, 64, 1, training)

    res3_1 = res_block(res2_2, 128, 2, training)
    res3_2 = res_block(res3_1, 128, 1, training)

    res4_1 = res_block(res3_2, 256, 2, training)
    res4_2 = res_block(res4_1, 256, 1, training)

    res5_1 = res_block(res4_2, 512, 2, training)
    res5_2 = res_block(res5_1, 512, 1, training)

    avg_pooling = tf.layers.max_pooling2d(res5_2, 7, 1, padding='same')

    flatten = tf.layers.flatten(avg_pooling)
    fc = tf.layers.dense(flatten, units=1000, activation=tf.nn.relu,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-5))
    return tf.layers.dense(fc, units=2)
