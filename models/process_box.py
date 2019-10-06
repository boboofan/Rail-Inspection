import tensorflow as tf


def clip_boxes(boxes, max_size):
    '''
    :param boxes: [N, 4]
    :param max_size: max_x, max_y
    '''
    boxes = tf.maximum(boxes, 0.0)

    max = tf.stack([max_size[0], max_size[1], max_size[0], max_size[1]])
    max = tf.cast(max, dtype=tf.float32)

    boxes = tf.minimum(boxes, max)

    return boxes


def boxes_area(boxes):
    min_x, min_y, max_x, max_y = tf.split(boxes, 4, axis=-1)
    return tf.squeeze((max_x - min_x) * (max_y - min_y), axis=-1)


def boxes_to_offsets(boxes, anchors):
    '''
    :param boxes: [..., 4] min_x, min_y, max_x, max_y
    :param anchors: [..., 4] min_x, min_y, max_x, max_y
    :return: [..., 4] dx, dy, dw, dh
    '''
    boxes_min_x, boxes_min_y, boxes_max_x, boxes_max_y = tf.split(boxes, 4, axis=-1)
    boxes_width = boxes_max_x - boxes_min_x
    boxes_height = boxes_max_y - boxes_min_y
    boxes_center_x = boxes_min_x + boxes_width * 0.5
    boxes_center_y = boxes_min_y + boxes_height * 0.5

    anchors_min_x, anchors_min_y, anchors_max_x, anchors_max_y = tf.split(anchors, 4, axis=-1)
    anchors_width = anchors_max_x - anchors_min_x
    anchors_height = anchors_max_y - anchors_min_y
    anchors_center_x = anchors_min_x + anchors_width * 0.5
    anchors_center_y = anchors_min_y + anchors_height * 0.5

    dx = (boxes_center_x - anchors_center_x) / anchors_width
    dy = (boxes_center_y - anchors_center_y) / anchors_height
    dw = tf.log(boxes_width / anchors_width)
    dh = tf.log(boxes_height / anchors_height)

    return tf.concat([dx, dy, dw, dh], axis=-1)


def offsets_to_boxes(offsets, anchors):
    '''
    :param offsets: [..., 4] dx, dy, dw, dh
    :param anchors: [..., 4] min_x, min_y, max_x, max_y
    :return: [..., 4] min_x, min_y, max_x, max_y
    '''
    dx, dy, dw, dh = tf.split(offsets, 4, axis=-1)

    anchors_min_x, anchors_min_y, anchors_max_x, anchors_max_y = tf.split(anchors, 4, axis=-1)
    anchors_width = anchors_max_x - anchors_min_x
    anchors_height = anchors_max_y - anchors_min_y
    anchors_center_x = anchors_min_x + anchors_width * 0.5
    anchors_center_y = anchors_min_y + anchors_height * 0.5

    boxes_width = tf.exp(dw) * anchors_width
    boxes_height = tf.exp(dh) * anchors_height
    boxes_center_x = dx * anchors_width + anchors_center_x
    boxes_center_y = dy * anchors_height + anchors_center_y

    boxes_min_x = boxes_center_x - boxes_width * 0.5
    boxes_min_y = boxes_center_y - boxes_height * 0.5
    boxes_max_x = boxes_center_x + boxes_width * 0.5
    boxes_max_y = boxes_center_y + boxes_height * 0.5

    return tf.concat([boxes_min_x, boxes_min_y, boxes_max_x, boxes_max_y], axis=-1)


def pairwise_intersection_area(boxes1, boxes2):
    '''
    :param boxes1: [N, 4]
    :param boxes2: [M, 4]
    :return: [N, M]
    '''

    min_x1, min_y1, max_x1, max_y1 = tf.split(boxes1, 4, axis=-1)  # [N, 1]
    min_x2, min_y2, max_x2, max_y2 = tf.split(boxes2, 4, axis=-1)  # [M, 1]

    minimun_max_y = tf.minimum(max_y1, tf.transpose(max_y2, [1, 0]))
    maximum_min_y = tf.maximum(min_y1, tf.transpose(min_y2, [1, 0]))
    intersecting_h = tf.maximum(0.0, minimun_max_y - maximum_min_y)  # [N, M]

    minimun_max_x = tf.minimum(max_x1, tf.transpose(max_x2, [1, 0]))
    maximum_min_x = tf.maximum(min_x1, tf.transpose(min_x2, [1, 0]))
    intersecting_w = tf.maximum(0.0, minimun_max_x - maximum_min_x)  # [N, M]

    return tf.multiply(intersecting_h, intersecting_w)  # [N, M]


def pairwise_iou(boxes1, boxes2):
    '''
    :param boxes1: [N, 4]
    :param boxes2: [M, 4]
    :return: [N, M]
    '''

    intersections = pairwise_intersection_area(boxes1, boxes2)  # [N, M]
    areas1 = boxes_area(boxes1)  # [N]
    areas2 = boxes_area(boxes2)  # [M]

    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections

    return tf.truediv(intersections, unions)
