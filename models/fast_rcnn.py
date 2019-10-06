import tensorflow as tf
from models.process_box import boxes_to_offsets, pairwise_iou


def sample_proposal_boxes(boxes, gt_boxes, gt_labels, boxes_num_per_image, foreground_thresh, foreground_ratio):
    '''
    :param boxes: [N, 4]
    :param gt_boxes: [M, 4]
    :param gt_labels: [M]
    '''

    iou = pairwise_iou(boxes, gt_boxes)  # [N, M]

    boxes = tf.concat([boxes, gt_boxes], axis=0)  # [N+M, 4]
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # [N+M, M]

    foreground_mask = tf.reduce_max(iou, axis=-1) > foreground_thresh
    foreground_index = tf.reshape(tf.where(foreground_mask), [-1])
    foreground_num = tf.minimum(tf.size(foreground_index), int(boxes_num_per_image * foreground_ratio))
    foreground_index = tf.random_shuffle(foreground_index)[:foreground_num]

    background_mask = tf.logical_not(foreground_mask)
    background_index = tf.reshape(tf.where(background_mask), [-1])
    background_num = tf.minimum(boxes_num_per_image - foreground_num, tf.size(background_index))
    background_index = tf.random_shuffle(background_index)[:background_num]

    foreground_gt_index = tf.gather(tf.argmax(iou, axis=-1), foreground_index)

    all_indices = tf.concat([foreground_index, background_index], axis=0)
    sampled_boxes = tf.gather(boxes, all_indices)
    sampled_labels = tf.concat(
        [tf.gather(gt_labels, foreground_gt_index), tf.zeros_like(background_index, dtype=tf.int32)], axis=0)

    return tf.stop_gradient(sampled_boxes), tf.stop_gradient(sampled_labels), tf.stop_gradient(foreground_gt_index)


def points_to_image(points):
    '''
    :param points: [N, 2]   x, y
    :return: [h, w, 1]
    '''

    points = tf.cast(tf.round(points), tf.int64)

    max_size = tf.reduce_max(points, axis=0)  # max_x, max_y

    x, y = tf.split(points, 2, axis=-1)
    tensor_indices = tf.concat([max_size[1] - y, x], axis=-1)
    tensor_indices = tf.sort(tensor_indices, axis=0)

    sparse_image = tf.sparse.SparseTensor(tensor_indices, values=tf.ones([tf.shape(points)[0]], dtype=tf.int64) * 255,
                                          dense_shape=max_size[::-1] + 1)
    image = tf.sparse.to_dense(sparse_image, default_value=0)

    return tf.cast(tf.expand_dims(image, axis=-1), dtype=tf.float32)


def pointwise_pooling_and_generate_image(points, crop_size):
    '''
    :param points: [N, 3]   x, y, channel
    :param crop_size: crop_height, crop_width
    :return: [crop_height, crop_width, 1]
    '''
    x, y, channel = tf.unstack(points, axis=-1)
    min_x, min_y, max_x, max_y = tf.reduce_min(x), tf.reduce_min(y), tf.reduce_max(x), tf.reduce_max(y)
    width, height = max_x - min_x, max_y - min_y
    crop_height, crop_width = crop_size

    stride_h = height / crop_height
    stride_w = width / crop_width

    range_h = tf.range(crop_height, dtype=tf.float32)
    range_w = tf.range(crop_width, dtype=tf.float32)

    intervals_h = tf.concat([tf.expand_dims(range_h * stride_h, axis=-1),
                             tf.expand_dims((range_h + 1) * stride_h, axis=-1)], axis=-1)
    intervals_w = tf.concat([tf.expand_dims(range_w * stride_w, axis=-1),
                             tf.expand_dims((range_w + 1) * stride_w, axis=-1)], axis=-1)

    intervals_h_size = tf.shape(intervals_h)[0]
    intervals_w_size = tf.shape(intervals_w)[0]

    intervals_h = tf.tile(tf.expand_dims(intervals_h, axis=1), [1, intervals_w_size, 1])
    intervals_w = tf.tile(tf.expand_dims(intervals_w, axis=0), [intervals_h_size, 1, 1])

    intervals = tf.concat([intervals_h, intervals_w], axis=-1)

    def get_channel(interval):
        assert interval.get_shape().as_list() == [4]

        mask = (interval[0] <= y) & (y <= interval[1]) & (interval[2] <= x) & (x <= interval[3])
        vaild_channel = tf.boolean_mask(channel, mask)

        return tf.cond(tf.size(vaild_channel) > 0,
                       lambda: tf.reduce_mean(vaild_channel),
                       lambda: tf.constant(0, dtype=tf.float32))

    flatten_image = tf.map_fn(get_channel, tf.reshape(intervals, [-1, 4]), dtype=tf.float32)
    image = tf.reshape(flatten_image, [intervals_h_size, intervals_w_size, 1])

    image = tf.image.flip_up_down(image)

    return tf.cast(image, dtype=tf.float32)


def crop_and_generate_images(points, boxes, crop_size):
    '''
    :param points: [N, 3]  x, y
    :param boxes: [M, 4]  min_x, min_y, max_x, max_y
    :param crop_size: crop_height, crop_width
    :return: [M, image_height, image_width, 1]
    '''

    def crop_points_and_generate_image(box):
        crop_mask = (box[0] <= points[:, 1]) & (points[:, 1] <= box[2]) & \
                    (box[1] <= points[:, 2]) & (points[:, 2] <= box[3])

        cropped_points = tf.boolean_mask(points, crop_mask)

        # cropped_image = points_to_image(cropped_points)
        # cropped_image = tf.image.resize_area(tf.expand_dims(cropped_image, axis=0), crop_size)
        #
        # return tf.squeeze(cropped_image, axis=0)

        return pointwise_pooling_and_generate_image(cropped_points, crop_size)

    images = tf.map_fn(crop_points_and_generate_image, boxes, dtype=tf.float32)
    # flip_lr = tf.image.flip_left_right(images)
    # flip_ud = tf.image.random_flip_up_down(images)

    return images


def roi_align(image, boxes, box_ind, crop_size):
    '''
    :param image: [1, h, w, 1]
    :param boxes: [N, 4]
    :param box_ind: [N]
    :param crop_size: height, width
    :return:[N, h, w, 1]
    '''

    # # TF's crop_and_resize produces zeros on border
    # if pad_border:
    #     # this can be quite slow
    #     image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
    #     boxes = boxes + 1

    image_shape = tf.shape(image)  # [4]
    image_height, image_width = tf.unstack(image_shape)[1:3]
    image_height, image_width = tf.cast(image_height, dtype=tf.float32), tf.cast(image_width, dtype=tf.float32)

    # map value of x, y to [0, 1]
    min_x, min_y, max_x, max_y = tf.split(boxes, 4, axis=-1)
    min_x = min_x / (image_width - 1.0)
    min_y = min_y / (image_height - 1.0)
    max_x = max_x / (image_width - 1.0)
    max_y = max_y / (image_height - 1.0)
    boxes = tf.concat([min_y, min_x, max_y, max_x], axis=-1)

    cropped_images = tf.image.crop_and_resize(image, boxes, box_ind, [int(crop_size[0] * 2), int(crop_size[1] * 2)])
    cropped_images = tf.layers.average_pooling2d(cropped_images, 2, 2, padding='same')

    return cropped_images


def fastrcnn_losses(boxes, labels, foreground_gt_index, offset_logits, label_logits, gt_boxes):
    '''
    :param boxes: [N, 4]
    :param labels: [N]
    :param foreground_gt_index: [foreground_num]
    :param offset_logits: [N, classes_num, 4]
    :param label_logits: [N, classes_num]
    :param gt_boxes: [M, 4]
    :return:
    '''

    # offset loss
    foreground_index = tf.reshape(tf.where(labels > 0), [-1])  # [foreground_num]
    foreground_num = tf.size(foreground_index)

    foreground_boxes = tf.gather(boxes, foreground_index)  # [foreground_num, 4]
    foreground_labels = tf.gather(labels, foreground_index)  # [foreground_num]

    foreground_gt_boxes = tf.gather(gt_boxes, foreground_gt_index)  # [foreground_num, 4]

    foreground_gt_offsets = boxes_to_offsets(foreground_gt_boxes, foreground_boxes)  # [foreground_num, 4]

    foreground_logit_indices = tf.concat([tf.range(foreground_num), foreground_labels], axis=-1)  # [foreground_num, 2]
    foreground_offset_logits = tf.gather_nd(offset_logits, foreground_logit_indices)  # [foreground_num, 4]

    offset_loss = tf.reduce_mean(tf.abs(foreground_gt_offsets - foreground_offset_logits))

    # label loss
    label_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss)

    # metrics
    train_accuary = tf.metrics.accuracy(labels=labels,
                                        predictions=tf.argmax(tf.nn.softmax(label_logits, axis=-1), axis=-1,
                                                              output_type=tf.int32))[1]

    return [offset_loss, label_loss], train_accuary


def fastrcnn_predictions(boxes, scores, score_threshold, boxes_num_per_image, nms_threshold):
    '''
    :param boxes: [N, classes_num + 1, 4]
    :param scores: [N, classes_num + 1]
    :return:
    '''
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # [classes_num, N, 4]
    scores = tf.transpose(scores[:, 1:], [1, 0])  # [classes_num, N]

    def get_mask(inputs):
        box, score = inputs  # [N, 4], [N]

        vaild_index = tf.reshape(tf.where(score > score_threshold), [-1])
        vaild_box = tf.gather(box, vaild_index)
        vaild_score = tf.gather(score, vaild_index)

        nms_index = tf.image.non_max_suppression(vaild_box, vaild_score, boxes_num_per_image, nms_threshold)

        vaild_index = tf.gather(vaild_index, nms_index)
        vaild_index = tf.sort(vaild_index)

        mask = tf.sparse.SparseTensor(indices=tf.expand_dims(vaild_index, axis=1),
                                      values=tf.ones_like(vaild_index, dtype=tf.bool),
                                      dense_shape=tf.shape(score, out_type=tf.int64))

        return tf.sparse.to_dense(mask, default_value=False)  # [N] bool

    vaild_masks = tf.map_fn(get_mask, [boxes, scores], dtype=tf.bool)  # [classes_num, N]

    vaild_indices = tf.where(vaild_masks)
    vaild_scores = tf.boolean_mask(scores, vaild_masks)

    topk_valid_scores, topk_indices = tf.nn.top_k(vaild_scores,
                                                  tf.minimum(boxes_num_per_image, tf.size(vaild_scores)),
                                                  sorted=False)
    topk_valid_indices = tf.gather(vaild_indices, topk_indices)  # [topk, 2]  classes_num's indices, boxes's indices
    classes_num_indices, _ = tf.unstack(topk_valid_indices, axis=-1)

    final_boxes = tf.gather_nd(boxes, topk_valid_indices)  # [topk, 4]
    final_scores = tf.identity(topk_valid_scores)  # [topk]
    final_labels = tf.add(classes_num_indices, 1)  # [topk]

    return final_boxes, final_labels, final_scores
