import tensorflow as tf

from models.region_proposal_network import RPN
from models.fast_rcnn import Fast_RCNN
from models.process_box import clip_boxes, offsets_to_boxes
from models.utils import get_chebyshev_polynomials


class Pointwise_RCNN(RPN, Fast_RCNN):
    def __init__(self, max_degree=1, regularizer=5e-5):
        super(Pointwise_RCNN, self).__init__(regularizer)

        self.max_degree = max_degree
        self.loss_lambda = 1
        self.ap_threshold = 0.3

    def __get_polynomials(self, points):
        # points: [points_num, 3]
        polynomials = tf.numpy_function(get_chebyshev_polynomials, [points, self.max_degree], tf.float32)

        points_shape = points.get_shape().as_list()
        polynomials.set_shape([self.max_degree + 1, points_shape[0], points_shape[0]])

        return polynomials

    def rpn(self, points, polynomials, gt_boxes, max_size, training):
        '''
        :param points: [points_num, 3]
        :param gt_boxes: [gt_boxes_num, 4]  min_x, min_y, max_x, max_y
        :param max_size: max_x, max_y
        '''
        anchors = self.generate_anchors(points)  # [points_num * anchors_num, 4]
        norm_points = self.normalization(points, batch=False)

        pred_offsets, pred_scores = self.rpn_head(norm_points, polynomials, self.anchors_num)
        pred_boxes = offsets_to_boxes(pred_offsets, anchors)

        proposal_boxes, proposal_scores = self.generate_rpn_proposals(pred_boxes, pred_scores, max_size, training)

        if training:
            reg_loss, cls_loss = self.rpn_losses(anchors, gt_boxes, pred_offsets, pred_scores)
        else:
            reg_loss, cls_loss = None, None

        return [proposal_boxes, proposal_scores], [reg_loss, cls_loss]

    def roi_heads(self, points, proposal_boxes, gt_boxes, gt_labels, max_size, classes_num, training):
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
            proposal_boxes, proposal_labels, foreground_gt_index = self.sample_proposal_boxes(proposal_boxes,
                                                                                              gt_boxes, gt_labels)
        else:
            proposal_boxes, proposal_labels, foreground_gt_index = proposal_boxes, None, None

        cropped_points = self.crop_and_sample(points, proposal_boxes)
        polynomials = tf.map_fn(self.__get_polynomials, cropped_points, tf.float32)
        cropped_points = self.normalization(cropped_points, batch=True)

        # [N, classes_num+1, 4], [N, classes_num+1]
        pred_offsets, pred_labels = self.extract_feature(cropped_points, polynomials, classes_num)

        if training:
            reg_loss, cls_loss, train_accuracy = self.fastrcnn_losses(proposal_boxes, proposal_labels,
                                                                      foreground_gt_index,
                                                                      pred_offsets, pred_labels, gt_boxes)
            return [reg_loss, cls_loss, train_accuracy]
        else:
            proposal_anchors = tf.tile(tf.expand_dims(proposal_boxes, axis=1), [1, classes_num + 1, 1])
            pred_boxes = offsets_to_boxes(pred_offsets, proposal_anchors)
            pred_boxes = clip_boxes(pred_boxes, max_size)  # [N, classes_num+1, 4]

            pred_scores = tf.nn.softmax(pred_labels, axis=-1)  # [N, classes_num+1]

            final_boxes, final_labels, final_scores = self.fastrcnn_predictions(pred_boxes, pred_scores)

            return [final_boxes, final_labels, final_scores]

    def build_network(self, points, polynomials, gt_boxes, gt_labels, max_size, classes_num, training):
        proposals, rpn_loss = self.rpn(points, polynomials, gt_boxes, max_size, training)
        outputs = self.roi_heads(points, proposals[0], gt_boxes, gt_labels, max_size, classes_num, training)
        if training:
            loss = rpn_loss[0] + outputs[0] + self.loss_lambda * (rpn_loss[1] + outputs[1])
            train_accuracy = outputs[2]

            return [loss, train_accuracy]
        else:
            return outputs
