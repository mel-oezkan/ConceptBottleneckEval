#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
import numpy as np
import os




def get_KP_BB(gt_point, mask_h, mask_w, bird_BB, KNOW_BIRD_BB=False):
    KP_best_x, KP_best_y = gt_point[0], gt_point[1]
    KP_x1 = KP_best_x - int(mask_w / 2)
    KP_x2 = KP_best_x + int(mask_w / 2)
    KP_y1 = KP_best_y - int(mask_h / 2)
    KP_y2 = KP_best_y + int(mask_h / 2)
    if KNOW_BIRD_BB:
        Bound = bird_BB
    else:
        Bound = [0, 0, 223, 223]
    if KP_x1 < Bound[0]:
        KP_x1, KP_x2 = Bound[0], Bound[0] + mask_w
    elif KP_x2 > Bound[2]:
        KP_x1, KP_x2 = Bound[2] - mask_w, Bound[2]
    if KP_y1 < Bound[1]:
        KP_y1, KP_y2 = Bound[1], Bound[1] + mask_h
    elif KP_y2 > Bound[3]:
        KP_y1, KP_y2 = Bound[3] - mask_h, Bound[3]
    return [KP_x1, KP_y1, KP_x2, KP_y2]#{'x1': KP_x1, 'x2': KP_x2, 'y1': KP_y1, 'y2': KP_y2}



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0 and iou <= 1.0
    return iou





