import numpy as np
import tensorflow as tf

from models.fusion_convolution import Graph_Convolution, Fusion_Convolution
from models.process_box import clip_boxes, boxes_to_offsets
from models.fast_rcnn import pairwise_iou


class RPN(Fusion_Convolution):
    def __init__(self, regularizer=5e-5):
        super(RPN, self).__init__(regularizer)

        self.anchor_sizes = [16, 32, 64]
        self.anchor_ratios = [0.1, 0.5, 1, 2, 5, 7]
        self.anchors_num = len(self.anchor_sizes) * len(self.anchor_ratios) * 8

        self.train_pre_nms_topk = 2000  # topk before nms
        self.train_post_nms_topk = 1000  # topk after nms
        self.test_pre_nms_topk = 1000
        self.test_post_nms_topk = 500
        self.min_edge_size = 0

        self.positive_anchor_threshold = 0.7
        self.negative_anchor_threshold = 0.3

        self.proposal_nms_threshold = 0.7

    def rpn_head(self, points, polynomials, anchors_num):
        '''
        :param points: [P, c]
        :param polynomials: [max_degree + 1, P, P]
        :param anchors_num: int
        :return: box_logits: [P * anchors_num, 4], label_logits: [P * anchors_num]
        '''

        batch_points, batch_polynomials = tf.expand_dims(points, axis=0), tf.expand_dims(polynomials, axis=0)

        fcu1 = self.fusion_convolution_unit('rpn_head_fuc1', batch_points, batch_polynomials, 48, activation=tf.nn.relu)
        fcu2 = self.fusion_convolution_unit('rpn_head_fuc2', batch_points, batch_polynomials, 48, activation=tf.nn.relu)
        concat = tf.concat([tf.expand_dims(fcu1, axis=-1), tf.expand_dims(fcu2, axis=-1)], axis=-1)

        head = tf.reduce_mean(concat, axis=-1)
        dropout = tf.layers.dropout(head)

        box_logits = self.graph_convolution(dropout, batch_polynomials, anchors_num * 4, name='rpn_head_box')
        label_logits = self.graph_convolution(dropout, batch_polynomials, anchors_num, name='rpn_head_label')

        return tf.reshape(box_logits, [-1, 4]), tf.reshape(label_logits, [-1])

    def generate_anchors(self, points):
        '''
        :param points: [points_num, 3] x, y, channel
        :return: [points_num * anchors_num, 4]
        '''

        points = points[:, :2]

        anchors = []
        for size in self.anchor_sizes:
            for ratio in self.anchor_ratios:
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

    def generate_rpn_proposals(self, boxes, scores, max_size, training):
        '''
        :param boxes: [N, 4]
        :param scores: [N]
        :return: proposal_boxes: [post_nms_topk, 4], proposal_scores: [post_nms_topk]
        '''

        topk = tf.minimum(tf.shape(boxes)[0], self.train_pre_nms_topk if training else self.test_pre_nms_topk)

        topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
        topk_boxes = tf.gather(boxes, topk_indices)
        topk_boxes = clip_boxes(topk_boxes, max_size)

        min_xy, max_xy = tf.split(topk_boxes, 2, axis=-1)
        wh = max_xy - min_xy
        valid_mask = tf.reduce_all(wh > self.min_edge_size, axis=-1)

        valid_boxes = tf.boolean_mask(topk_boxes, valid_mask)
        valid_scores = tf.boolean_mask(topk_scores, valid_mask)

        min_yx, max_yx = tf.reverse(min_xy, axis=[-1]), tf.reverse(max_xy, axis=[-1])
        valid_boxes_reversed = tf.boolean_mask(tf.concat([min_yx, max_yx], axis=-1), valid_mask)
        nms_indices = tf.image.non_max_suppression(valid_boxes_reversed, valid_scores,
                                                   self.train_post_nms_topk if training else self.test_post_nms_topk,
                                                   self.proposal_nms_threshold)

        proposal_boxes = tf.stop_gradient(tf.gather(valid_boxes, nms_indices))  # [post_nms_topk, 4]
        proposal_scores = tf.stop_gradient(tf.gather(valid_scores, nms_indices))  # [post_nms_topk]

        return proposal_boxes, proposal_scores

    def rpn_losses(self, anchors, gt_boxes, pred_offsets, pred_labels):
        '''
        :param anchors: [N, 4]
        :param gt_boxes: [M, 4]
        :param pred_offsets: [N, 4]
        :param pred_labels: [N]
        :return:
        '''

        iou = pairwise_iou(anchors, gt_boxes)  # [N, M]

        positive_mask = tf.cond(tf.shape(iou)[1] > 0,
                                lambda: tf.reduce_max(iou, axis=1) > self.positive_anchor_threshold,
                                lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))
        negative_mask = tf.cond(tf.shape(iou)[1] > 0,
                                lambda: tf.reduce_max(iou, axis=1) < self.negative_anchor_threshold,
                                lambda: tf.ones([tf.shape(iou)[0]], dtype=tf.bool))

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
        valid_mask = positive_mask | negative_mask

        valid_pred_labels = tf.boolean_mask(pred_labels, valid_mask)

        gt_labels = tf.where(positive_mask,
                             tf.ones_like(pred_labels, dtype=tf.float32),
                             tf.zeros_like(pred_labels, dtype=tf.float32))
        valid_gt_labels = tf.boolean_mask(gt_labels, valid_mask)

        cls_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_gt_labels, logits=valid_pred_labels))

        return reg_loss, cls_loss
