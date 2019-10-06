import tensorflow as tf


def conv1d(inputs, filters=None, kernel_size=3, strides=1, padding='same', activation=None):
    if not filters:
        filters = inputs.get_shape().as_list()[-1]
    return tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def conv2d(inputs, filters=None, kernel_size=3, strides=1, padding='same', activation=None):
    if not filters:
        filters = inputs.get_shape().as_list()[-1]
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def max_pooling1d(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling1d(inputs, pool_size, strides, padding)


def max_pooling2d(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def res_block(inputs, output_dim, strides, is_point, training):
    input_dim = inputs.get_shape().as_list()[-1]

    if input_dim == output_dim:
        if strides == 2:
            shortcut = max_pooling1d(inputs) if is_point else max_pooling2d(inputs)
        else:
            shortcut = inputs
    else:
        shortcut = conv1d(inputs, output_dim, 1, strides) if is_point else conv2d(inputs, output_dim, 1, strides)

    conv1 = conv1d(inputs, output_dim, strides=strides) if is_point else conv2d(inputs, output_dim, strides=strides)
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    conv2 = conv1d(relu1, output_dim) if is_point else conv2d(relu1, output_dim)
    bn2 = tf.layers.batch_normalization(conv2, training=training)
    relu2 = tf.nn.relu(bn2)

    return tf.add(shortcut, relu2)


def extract_points_feature(points, training):
    '''
    :param points: [N, 3]
    :param training:
    :return:
    '''
    points = tf.expand_dims(points, axis=0)  # [1, N, 3]

    res1 = res_block(points, 16, 1, True, training)
    pool1 = max_pooling1d(res1, strides=1)

    res2 = res_block(pool1, 32, 1, True, training)
    pool2 = max_pooling1d(res2, strides=1)

    res3 = res_block(pool2, 64, 1, True, training)
    pool3 = max_pooling1d(res3, strides=1)

    # res4 = res_block(pool3, 128, 1, True, training)
    # pool4 = max_pooling1d(res4, strides=1)
    #
    # res5 = res_block(pool4, 256, 1, True, training)
    # pool5 = max_pooling1d(res5, strides=1)

    return tf.squeeze(pool3, axis=0)  # [N, 256]


def extract_images_feature(images, training):
    '''
    :param images: [N, h, w, 1]
    :param training: bool
    :return:
    '''
    conv1 = conv2d(images, 64, 7, 2)
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    pool1 = tf.layers.max_pooling2d(relu1, 3, 2, 'same')
    res2_1 = res_block(pool1, 64, 1, False, training)
    res2_2 = res_block(res2_1, 64, 1, False, training)

    res3_1 = res_block(res2_2, 128, 2, False, training)
    res3_2 = res_block(res3_1, 128, 1, False, training)

    res4_1 = res_block(res3_2, 256, 2, False, training)
    res4_2 = res_block(res4_1, 256, 1, False, training)

    res5_1 = res_block(res4_2, 512, 2, False, training)
    res5_2 = res_block(res5_1, 512, 1, False, training)

    return res5_2
