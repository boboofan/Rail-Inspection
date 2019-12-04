import tensorflow as tf

from models.process_box import boxes_to_offsets, pairwise_iou
from models.fusion_convolution import Fusion_Convolution
from models.utils import crop_and_sample_points


class Fast_RCNN(Fusion_Convolution):
    def __init__(self, regularizer=5e-5):
        super(Fast_RCNN, self).__init__(regularizer)

        self.boxes_num_per_image = 600
        self.foreground_threshold = 0.5
        self.foreground_ratio = 0.25
        self.points_num_each_box = 150
        self.score_threshold = 0.05
        self.test_boxes_num_per_image = 100
        self.test_nms_threshold = 0.5

    def sample_proposal_boxes(self, boxes, gt_boxes, gt_labels):
        '''
        :param boxes: [N, 4]
        :param gt_boxes: [M, 4]
        :param gt_labels: [M]
        '''

        iou = pairwise_iou(boxes, gt_boxes)  # [N, M]

        boxes = tf.concat([boxes, gt_boxes], axis=0)  # [N+M, 4]
        iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # [N+M, M]

        foreground_mask = tf.cond(tf.shape(iou)[1] > 0,
                                  lambda: tf.reduce_max(iou, axis=-1) > self.foreground_threshold,
                                  lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))
        foreground_index = tf.reshape(tf.where(foreground_mask), [-1])
        foreground_num = tf.minimum(tf.size(foreground_index), int(self.boxes_num_per_image * self.foreground_ratio))
        foreground_index = tf.random_shuffle(foreground_index)[:foreground_num]

        background_mask = tf.logical_not(foreground_mask)
        background_index = tf.reshape(tf.where(background_mask), [-1])
        background_num = tf.minimum(self.boxes_num_per_image - foreground_num, tf.size(background_index))
        background_index = tf.random_shuffle(background_index)[:background_num]

        highest_iou_gt_index = tf.cond(tf.shape(iou)[1] > 0,
                                       lambda: tf.argmax(iou, axis=-1, output_type=tf.int32),
                                       lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.int32))
        foreground_gt_index = tf.gather(highest_iou_gt_index, foreground_index)

        all_indices = tf.concat([foreground_index, background_index], axis=0)
        sampled_boxes = tf.gather(boxes, all_indices)
        sampled_labels = tf.concat(
            [tf.gather(gt_labels, foreground_gt_index), tf.zeros_like(background_index, dtype=tf.int32)], axis=0)

        return tf.stop_gradient(sampled_boxes), tf.stop_gradient(sampled_labels), tf.stop_gradient(foreground_gt_index)

    def crop_and_sample(self, points, boxes):
        cropped_points = tf.numpy_function(
            crop_and_sample_points, [points, boxes, self.points_num_each_box], tf.float32)

        shape = [boxes.get_shape().as_list()[0], self.points_num_each_box, points.get_shape().as_list()[-1]]
        cropped_points.set_shape(shape)

        return cropped_points

    def extract_feature(self, inputs, polynomials, classes_num):
        fcu1 = self.fusion_convolution_unit('backbone_fuc1', inputs, polynomials, 512, activation=tf.nn.relu)
        gmp1, gvp1 = self.global_max_pooling(fcu1), self.global_variance_pooling(fcu1)
        concat1 = tf.concat([gmp1, gvp1], axis=-1)

        fcu2 = self.fusion_convolution_unit('backbone_fuc2', inputs, polynomials, 512, activation=tf.nn.relu)
        gmp2, gvp2 = self.global_max_pooling(fcu2), self.global_variance_pooling(fcu2)
        concat2 = tf.concat([gmp2, gvp2], axis=-1)

        concat3 = tf.concat([concat1, concat2], axis=1)

        flatten = tf.layers.flatten(concat3)
        dropout = tf.layers.dropout(flatten)

        offset_logits = self.dense(dropout, units=(classes_num + 1) * 4)
        offset_logits = tf.reshape(offset_logits, [-1, classes_num + 1, 4])

        label_logits = self.dense(dropout, units=classes_num + 1, activation=tf.nn.softmax)

        return offset_logits, label_logits

    def fastrcnn_losses(self, boxes, labels, foreground_gt_index, pred_offsets, pred_labels, gt_boxes):
        '''
        :param boxes: [N, 4]
        :param labels: [N]
        :param foreground_gt_index: [foreground_num]
        :param pred_offsets: [N, classes_num, 4]
        :param pred_labels: [N, classes_num]
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

        foreground_logit_indices = tf.concat([tf.range(foreground_num), foreground_labels],
                                             axis=-1)  # [foreground_num, 2]
        foreground_offset_logits = tf.gather_nd(pred_offsets, foreground_logit_indices)  # [foreground_num, 4]

        reg_loss = tf.reduce_mean(tf.abs(foreground_gt_offsets - foreground_offset_logits))
        reg_loss = tf.truediv(reg_loss, tf.cast(tf.shape(labels)[0], tf.float32), name='box_loss')

        # label loss
        cls_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=pred_labels)
        cls_loss = tf.reduce_mean(cls_loss, name='label_loss')

        # metrics
        predictions = tf.argmax(tf.nn.softmax(pred_labels), axis=-1, output_type=tf.int32)
        train_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)[1]

        return reg_loss, cls_loss, train_accuracy

    def fastrcnn_predictions(self, boxes, scores):
        '''
        :param boxes: [N, classes_num + 1, 4]
        :param scores: [N, classes_num + 1]
        :return:
        '''
        boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # [classes_num, N, 4]
        scores = tf.transpose(scores[:, 1:], [1, 0])  # [classes_num, N]

        def get_mask(inputs):
            box, score = inputs  # [N, 4], [N]

            valid_index = tf.reshape(tf.where(score > self.score_threshold), [-1])
            valid_box = tf.gather(box, valid_index)
            valid_score = tf.gather(score, valid_index)

            nms_index = tf.image.non_max_suppression(valid_box, valid_score, self.test_boxes_num_per_image,
                                                     self.test_nms_threshold)

            valid_index = tf.gather(valid_index, nms_index)
            valid_index = tf.sort(valid_index)

            mask = tf.sparse.SparseTensor(indices=tf.expand_dims(valid_index, axis=1),
                                          values=tf.ones_like(valid_index, dtype=tf.bool),
                                          dense_shape=tf.shape(score, out_type=tf.int64))

            return tf.sparse.to_dense(mask, default_value=False)  # [N] bool

        valid_masks = tf.map_fn(get_mask, [boxes, scores], dtype=tf.bool)  # [classes_num, N]

        valid_indices = tf.where(valid_masks)
        valid_scores = tf.boolean_mask(scores, valid_masks)

        topk_valid_scores, topk_indices = tf.nn.top_k(valid_scores,
                                                      tf.minimum(self.test_boxes_num_per_image, tf.size(valid_scores)),
                                                      sorted=False)
        topk_valid_indices = tf.gather(valid_indices, topk_indices)  # [topk, 2]  classes_num's indices, boxes's indices
        classes_num_indices, _ = tf.unstack(topk_valid_indices, axis=-1)

        final_boxes = tf.gather_nd(boxes, topk_valid_indices)  # [topk, 4]
        final_scores = tf.identity(topk_valid_scores)  # [topk]
        final_labels = tf.identity(classes_num_indices)  # [topk]

        return final_boxes, final_labels, final_scores
