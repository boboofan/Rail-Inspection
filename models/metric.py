import numpy as np


def is_TP(box, gt_boxes, threshold):
    '''
    :param box: [4] min_x, min_y, max_x, max_y
    :param gt_boxes: [M, 4] min_x, min_y, max_x, max_y
    '''

    maximum_min_x = np.maximum(box[0], gt_boxes[:, 0])
    minimun_max_x = np.minimum(box[2], gt_boxes[:, 2])
    intersecting_w = np.maximum(0.0, minimun_max_x - maximum_min_x)

    maximum_min_y = np.maximum(box[1], gt_boxes[:, 1])
    minimun_max_y = np.minimum(box[3], gt_boxes[:, 3])
    intersecting_h = np.maximum(0.0, minimun_max_y - maximum_min_y)

    intersecting_area = np.multiply(intersecting_h, intersecting_w)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    iou = np.true_divide(intersecting_area, box_area + gt_boxes_area - intersecting_area)  # [M]

    return np.max(iou) > threshold


def average_precision(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, threshold):
    '''
    :param pred_boxes: [N, 4]
    :param pred_labels: [N]
    :param pred_scores: [N]
    :param gt_boxes: [M, 4]
    :param gt_labels: [M]
    :return:
    '''

    TPs = []
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        pairwise_gt_boxes = gt_boxes[np.where(gt_labels == pred_labels[i])]
        TPs.append(is_TP(box, pairwise_gt_boxes, threshold))

    TPs = np.array(TPs)
    TPs = TPs[np.argsort(pred_scores)][::-1]

    acc_TP, acc_FP = 0, 0
    gt_len = len(gt_boxes)

    precisions_recalls = []
    for i in range(len(TPs)):
        if TPs[i]:
            acc_TP += 1
        else:
            acc_FP += 1

        precisions_recalls.append([acc_TP / (acc_TP + acc_FP), acc_TP / gt_len])

    precisions_recalls.sort(key=lambda x: x[0], reverse=True)

    AP = precisions_recalls[0][0] * precisions_recalls[0][1]
    pre_recall = precisions_recalls[0][1]
    for i in range(1, len(precisions_recalls)):
        precision, recall = precisions_recalls[i]
        if recall > pre_recall:
            AP += precision * (recall - pre_recall)
            pre_recall = recall

    return AP

