import numpy as np
import tensorflow as tf

from models.extract_feature import conv1d
from models.process_box import clip_boxes, boxes_to_offsets
from models.fast_rcnn import pairwise_iou


def rpn_head(feature, anchors_num):
    '''
    :param feature: [N, c]
    :param anchors_num: int
    :return: box_logits: [N * anchors_num, 4], label_logits: [N * anchors_num]
    '''

    head = conv1d(inputs=tf.expand_dims(feature, axis=0), activation=tf.nn.relu)

    box_logits = conv1d(inputs=head, filters=anchors_num * 4)
    label_logits = conv1d(inputs=head, filters=anchors_num)

    return tf.reshape(box_logits, [-1, 4]), tf.reshape(label_logits, [-1])


def generate_anchors(points, sizes, ratios):
    '''
    :param points: [points_num, 3] tf.float32
    :return: [points_num * anchors_num, 4]
    '''

    points = points[:, 1:]

    anchors = []
    for size in sizes:
        for ratio in ratios:
            # w*h = area, h/w = ratio
            w = np.sqrt(size * size / ratio)  # the width of anchors
            h = w * ratio  # the height of anchors

            # origin point is on the corners of anchors
            anchors += [[0, 0, w, h], [-w, -h, 0, 0], [-w, 0, 0, h], [0, -h, w, 0]]
            # origin point is on the middle of the edges of anchors
            anchors += [[-w / 2, -h, w / 2, 0], [-w / 2, 0, w / 2, h],
                        [0, -h / 2, w, h / 2], [-w, -h / 2, 0, h / 2]]

    # anchors_num = len(sizes) * len(ratios) * 8
    anchors = tf.constant(anchors, dtype=tf.float32)  # [anchors_num, 4]

    expand_points = tf.expand_dims(tf.tile(points, [1, 2]), axis=1)  # [points_num, 1, 4]
    expand_anchors = tf.expand_dims(anchors, axis=0)  # [1, anchors_num, 4]

    return tf.reshape(expand_points + expand_anchors, [-1, 4])  # [points_num * anchors_num, 4]


def generate_rpn_proposals(boxes, scores, pre_nms_topk, post_nms_topk,
                           proposal_nms_threshold, max_size, min_edge_size):
    '''
    :param boxes: [N, 4]
    :param scores: [N]
    :return: proposal_boxes: [post_nms_topk, 4], proposal_scores: [post_nms_topk]
    '''

    topk = tf.minimum(tf.shape(boxes)[0], pre_nms_topk)

    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, max_size)

    min_xy, max_xy = tf.split(topk_boxes, 2, axis=-1)
    wh = max_xy - min_xy
    vaild_mask = tf.reduce_all(wh > min_edge_size, axis=-1)

    vaild_boxes = tf.boolean_mask(topk_boxes, vaild_mask)
    vaild_scores = tf.boolean_mask(topk_scores, vaild_mask)

    min_yx, max_yx = tf.reverse(min_xy, axis=[-1]), tf.reverse(max_xy, axis=[-1])
    vaild_boxes_reversed = tf.boolean_mask(tf.concat([min_yx, max_yx], axis=-1), vaild_mask)
    nms_indices = tf.image.non_max_suppression(vaild_boxes_reversed, vaild_scores,
                                               post_nms_topk, proposal_nms_threshold)

    proposal_boxes = tf.stop_gradient(tf.gather(vaild_boxes, nms_indices))  # [post_nms_topk, 4]
    proposal_scores = tf.stop_gradient(tf.gather(vaild_scores, nms_indices))  # [post_nms_topk]

    return tf.stop_gradient(proposal_boxes, name='proposal_boxes'), tf.stop_gradient(proposal_scores,
                                                                                     name='proposal_scores')


def rpn_losses(anchors, gt_boxes, pred_offsets, pred_labels, positive_anchor_threshold, negative_anchor_threshold):
    '''
    :param anchors: [N, 4]
    :param gt_boxes: [M, 4]
    :param pred_offsets: [N, 4]
    :param pred_labels: [N]
    :return:
    '''

    iou = pairwise_iou(anchors, gt_boxes)  # [N, M]

    positive_mask = tf.reduce_max(iou, axis=1) > positive_anchor_threshold
    negative_mask = tf.reduce_max(iou, axis=1) < negative_anchor_threshold

    # reg loss
    positive_anchors = tf.boolean_mask(anchors, positive_mask)

    positive_gt_boxes_index = tf.argmax(tf.boolean_mask(iou, positive_mask), axis=-1)
    positive_gt_boxes = tf.gather(gt_boxes, positive_gt_boxes_index)
    positive_gt_offsets = boxes_to_offsets(positive_gt_boxes, positive_anchors)

    positive_pred_offsets = tf.boolean_mask(pred_offsets, positive_mask)

    delta = 1 / 9
    reg_loss = tf.losses.huber_loss(positive_gt_offsets, positive_pred_offsets, delta=delta,
                                    reduction=tf.losses.Reduction.SUM) / delta

    # cls loss
    vaild_mask = positive_mask | negative_mask

    vaild_pred_scores = tf.boolean_mask(pred_labels, vaild_mask)

    gt_labels = tf.where(positive_mask,
                         tf.ones_like(pred_labels, dtype=tf.float32),
                         tf.zeros_like(pred_labels, dtype=tf.float32))
    vaild_gt_labels = tf.boolean_mask(gt_labels, vaild_mask)

    cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=vaild_gt_labels, logits=vaild_pred_scores))

    return [reg_loss, cls_loss]
