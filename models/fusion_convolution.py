import tensorflow as tf


class Graph_Convolution:
    def __init__(self, name, input_dim, output_dim, activation=None, regularizer=None, use_bias=False):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.regularizer = regularizer
        self.use_bias = use_bias

    def get_weight(self, shape, name):
        with tf.variable_scope(self.name):
            return tf.get_variable(name=name, shape=shape, dtype=tf.float32, trainable=True,
                                   initializer=tf.glorot_uniform_initializer(),
                                   regularizer=tf.keras.regularizers.l2(self.regularizer))

    def get_bias(self, shape, name):
        with tf.variable_scope(self.name):
            return tf.get_variable(name=name, shape=shape, dtype=tf.float32, trainable=True,
                                   initializer=tf.zeros_initializer())

    def gcn(self, inputs):
        features, polynomials = inputs
        polynomials_list = tf.unstack(polynomials)
        outputs = []
        for i in range(len(polynomials_list)):
            weight = self.get_weight([self.input_dim, self.output_dim], name='weight' + str(i))
            outputs.append(tf.matmul(tf.matmul(polynomials_list[i], features), weight))

        output = tf.add_n(outputs)
        if self.use_bias:
            output = tf.add(output, self.get_bias([self.output_dim], name='bias'))

        if self.activation:
            output = self.activation(output)

        return output

    def __call__(self, features, polynomials):
        return tf.map_fn(self.gcn, [features, polynomials], dtype=tf.float32)


class Fusion_Convolution:
    def __init__(self, regularizer=5e-5, data_format='channels_last'):
        self.regularizer = regularizer
        self.data_format = data_format

    def normalization(self, points, gamma=1, beta=0, eps=1e-4, batch=True):
        mean, variance = tf.nn.moments(points, axes=[0, 1] if batch else [0])
        norm = (points - mean) / (tf.sqrt(variance) + eps)
        norm = gamma * norm + beta

        return norm

    def conv1d(self, inputs, filters, kernel_size=3, strides=1, padding='same', activation=None, name=None):
        return tf.layers.conv1d(inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, data_format=self.data_format, activation=activation,
                                kernel_regularizer=tf.keras.regularizers.l2(self.regularizer), name=name)

    def dense(self, inputs, units, activation=None, name=None):
        return tf.layers.dense(inputs, units=units, activation=activation,
                               kernel_regularizer=tf.keras.regularizers.l2(self.regularizer), name=name)

    def global_max_pooling(self, inputs):
        return tf.reduce_max(inputs, axis=1 if self.data_format == 'channel_lasts' else 2, keepdims=True)

    def global_variance_pooling(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=1 if self.data_format == 'channel_lasts' else 2, keepdims=True)
        return variance

    def graph_convolution(self, inputs, polynomials, output_dim, activation=None, name='gcn'):
        input_dim = inputs.get_shape().as_list()[-1]
        return Graph_Convolution(name=name, input_dim=input_dim, output_dim=output_dim,
                                 activation=activation, regularizer=self.regularizer)(inputs, polynomials)

    def fusion_convolution_unit(self, name, inputs, polynomials, output_dim, layers_dim=(48, 96), activation=None):
        with tf.variable_scope(name):
            conv1 = self.conv1d(inputs, filters=layers_dim[0], activation=activation, name='conv1')
            conv2 = self.conv1d(conv1, filters=layers_dim[1], activation=activation, name='conv2')
            dropout = tf.layers.dropout(conv2, name='dropout')
            gcn = self.graph_convolution(dropout, polynomials, output_dim=output_dim, activation=activation, name='gcn')

            return gcn
