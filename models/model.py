import tensorflow as tf

from models.extract_feature import extract_points_feature, extract_images_feature
from models.region_proposal_network import rpn_head, generate_anchors, generate_rpn_proposals, rpn_losses
from models.process_box import clip_boxes, offsets_to_boxes
from models.fast_rcnn import sample_proposal_boxes, fastrcnn_losses, fastrcnn_predictions, crop_and_generate_images


class Faster_RCNN:
    def __init__(self, classes_num):
        # backbone
        self.classes_num = classes_num
        self.scale_ratio = 1 / 32
        self.loss_lambda = 1

        # rpn
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

        # fast rcnn
        self.boxes_num_per_image = 600
        self.foreground_threshold = 0.5
        self.foreground_ratio = 0.25
        self.crop_size = [224, 224]

        # test
        self.score_threshold = 0.05
        self.test_boxes_num_per_image = 100
        self.test_nms_threshold = 0.5
        self.ap_threshold = 0.3

    def rpn(self, points, gt_boxes, max_size, training):
        '''
        :param points: [points_num, 3]
        :param gt_boxes: [gt_boxes_num, 4]  min_x, min_y, max_x, max_y
        :param max_size: max_height, max_width
        '''
        anchors = generate_anchors(points, self.anchor_sizes, self.anchor_ratios)  # [points_num*anchors_num, 4]

        points_feature = extract_points_feature(points, training)
        pred_offsets, pred_scores = rpn_head(points_feature, self.anchors_num)  # [n, 4], [n]
        pred_boxes = offsets_to_boxes(pred_offsets, anchors)  # [n, 4]

        self.proposal_boxes, self.proposal_scores = generate_rpn_proposals(
            pred_boxes, pred_scores,
            self.train_pre_nms_topk if training else self.test_pre_nms_topk,
            self.train_post_nms_topk if training else self.test_post_nms_topk,
            self.proposal_nms_threshold, max_size, self.min_edge_size)

        if training:
            self.rpn_losses = rpn_losses(anchors, gt_boxes, pred_offsets, pred_scores,
                                         self.positive_anchor_threshold, self.negative_anchor_threshold)

    def roi_heads(self, points, proposal_boxes, gt_boxes, gt_labels, max_size, training):
        '''
        :param points: [points_num, 3]
        :param proposal_boxes: [N, 4]
        :param gt_boxes: [gt_boxes_num, 4]
        :param gt_labels: [gt_boxes_num]
        :param max_size: max_x, max_y
        :param training: bool
        :return:
        '''

        if training:
            proposal_boxes, proposal_labels, foreground_gt_index = sample_proposal_boxes(proposal_boxes,
                                                                                         gt_boxes, gt_labels,
                                                                                         self.boxes_num_per_image,
                                                                                         self.foreground_threshold,
                                                                                         self.foreground_ratio)
        else:
            proposal_boxes, proposal_labels, foreground_gt_index = proposal_boxes, None, None

        cropped_image = crop_and_generate_images(points, proposal_boxes, self.crop_size)

        image_feature = extract_images_feature(cropped_image, training)  # [N, h, w, 512]
        gap = tf.reduce_mean(image_feature, axis=[1, 2])  # [N, 512] Global average pooling

        # [N, classes_num+1, 4]
        offset_logits = tf.layers.dense(gap, (self.classes_num + 1) * 4,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        offset_logits = tf.reshape(offset_logits, [-1, self.classes_num + 1, 4])

        # [N, classes_num+1]
        label_logits = tf.layers.dense(gap, self.classes_num + 1,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        if training:
            self.fastrcnn_losses, self.fastrcnn_accuracies = fastrcnn_losses(proposal_boxes, proposal_labels,
                                                                             foreground_gt_index,
                                                                             offset_logits, label_logits, gt_boxes)
        else:
            proposal_anchors = tf.tile(tf.expand_dims(proposal_boxes, axis=1), [1, self.classes_num + 1, 1])
            box_logits = offsets_to_boxes(offset_logits, proposal_anchors)
            box_logits = clip_boxes(box_logits, max_size)  # [N, classes_num+1, 4]

            label_scores = tf.nn.softmax(label_logits, axis=-1)  # [N, classes_num+1]

            self.final_boxes, self.final_labels, self.final_scores = fastrcnn_predictions(box_logits, label_scores,
                                                                                          self.score_threshold,
                                                                                          self.test_boxes_num_per_image,
                                                                                          self.test_nms_threshold)

    def get_loss(self, points, gt_boxes, gt_labels, max_size):
        self.rpn(points, gt_boxes, max_size, True)
        self.roi_heads(points, self.proposal_boxes, gt_boxes, gt_labels, max_size, True)

        loss = self.rpn_losses + self.fastrcnn_losses
        loss = loss[0] * self.loss_lambda + loss[1]
        return tf.reduce_mean(loss)

    def get_outputs(self, points, gt_boxes, gt_labels, max_size):
        self.rpn(points, gt_boxes, max_size, False)
        self.roi_heads(points, self.proposal_boxes, gt_boxes, gt_labels, max_size, False)

        return self.final_boxes, self.final_labels, self.final_scores + self.proposal_scores

    def get_train_accuracy(self, points, gt_boxes, gt_labels, max_size):
        self.rpn(points, gt_boxes, max_size, True)
        self.roi_heads(points, self.proposal_boxes, gt_boxes, gt_labels, max_size, True)

        return self.fastrcnn_accuracies
