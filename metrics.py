import numpy as np


def iou_soft(preds, gts):
    print(" [**] Calculating IoU", flush=True)
    assignments = []
    for x in range(len(preds)):
        cur_pred, cur_gt = preds[x], gts[x]
        stacked = np.stack([cur_pred, cur_gt], axis=-1)
        assignments.append(np.reshape(stacked, newshape=[-1, 2]))
    assignments = np.concatenate(assignments, axis=0)
    sorted_idxs = np.argsort(assignments[:, 0])[::-1]
    sorted_assignments = assignments[sorted_idxs]

    num_positives = np.sum(sorted_assignments[:, 1])

    # calculate tps and fps for each threshold
    tps = np.cumsum(sorted_assignments[:, 1], dtype=np.int)
    fps = np.cumsum(1 - sorted_assignments[:, 1], dtype=np.int)

    # union is simply all fps + all positives
    union = num_positives + fps

    # intersection is tp
    intersection = tps

    # calculate intersection over union
    inter_over_union = intersection / union
    curve = np.stack([sorted_assignments[:, 0], inter_over_union], axis=-1)
    best_thresh = curve[np.argmax(curve[:, 1])]

    print(" [**] Best IoU {} at {}".format(best_thresh[1], best_thresh[0]), flush=True)
    out = {"IoU": best_thresh[1],
           "threshold": best_thresh[0]}
    return out


def iou(preds, gts):
    print(" [**] Calculating IoU", flush=True)
    assignments = []
    for x in range(len(preds)):
        cur_pred, cur_gt = preds[x], gts[x]
        stacked = np.stack([cur_pred, cur_gt], axis=-1)
        assignments.append(np.reshape(stacked, newshape=[-1, 2]))
    assignments = np.concatenate(assignments, axis=0)

    classes = np.unique(assignments[:, 1])

    intersections = []
    unions = []
    for x in range(classes.shape[0]):
        pred_class_x = assignments[:, 0] == classes[x]
        gt_class_x = assignments[:, 1] == classes[x]
        intersection = np.sum(np.logical_and(pred_class_x, gt_class_x))
        intersections.append(intersection)
        union = np.sum(np.logical_or(pred_class_x, gt_class_x))
        unions.append(union)

    intersection_over_union = np.array(intersections) / np.array(unions)

    out = {}
    for x in range(intersection_over_union.shape[0]):
        out["IoU/class_{}".format(x)] = intersection_over_union[x]

    out["IoU/mean"] = np.mean(intersection_over_union)
    return out
