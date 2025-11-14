#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
import numpy as np
import os
import cv2

def calculate_atten_IoU(input, impath, save_att_idx, maps, names, vis_groups, KP_root=None, save_att=False, scale=4,
                        resize_WH=False, KNOW_BIRD_BB=False):
    """
    :param input: input image
    :param impath: image paths
    :param maps: maps with size 64, 312, 7, 7
    :param names: layer names
    :param vis_groups: vis_groups is a list of size image_num. each item is a dict,
                       including the attention index for each subgroup
    :param KP_root: is the root of KP_centers
    :return:
    """
    # print("resize_WH:{}, out_of_edge:{}, max_area_center:{}".format(resize_WH, out_of_edge, max_area_center))
    img_raw_show = tensor_imshow(input)  # (64, 3, 224, 224)
    attri_names = refine_attri_names('./data/vis/files/attri_name.txt')
    batch_IoU = []
    for i, path in enumerate(impath):
        # time_1 = time.time()
        # print(i, path)
        if type(vis_groups) is dict:
            vis_group = vis_groups
        elif type(vis_groups) is list:
            vis_group = vis_groups[i]
        else:
            exit('ERROR FOR vis_groups!')
        tmp = path.split('/')[-2:]
        this_dir = os.path.join(KP_root, tmp[0], tmp[1][:-4])
        if save_att:
            save_dir = os.path.join(save_att, tmp[0], tmp[1][:-4])
        else:
            save_dir = False
        # KP_BBs, bird_BB = read_BB(this_dir)
        KPs, bird_BB = read_KP(this_dir)
        # print("this dir:", this_dir)
        # time_2 = time.time()
        # print('time for load BB:', time_2 - time_1)
        img_IoU = get_IoU_Image(i, img_raw_show, maps, save_dir, save_att_idx, names, vis_group, attri_names,
                                KPs, bird_BB, scale, resize_WH, KNOW_BIRD_BB)
        # time_3 = time.time()
        # print('time for calculate IoU:', time_3 - time_2)
        batch_IoU.append(img_IoU)
        # exit()
    return batch_IoU



def get_IoU_Image(idx, imgs, maps, save_dir, save_att_idx, names, groups,
                  attri_names, KPs, bird_BB, scale, resize_WH, KNOW_BIRD_BB):
    BB_parts = ['head', 'breast', 'belly', 'back', 'wing', 'tail', 'leg']
    # {'x1': KP_x1, 'x2': KP_x2, 'y1': KP_y1, 'y2': KP_y2}
    attri_to_group_dict = {
        'head': [1, 4, 5, 6, 9, 10],
        'breast': [3, 14],
        'belly': [2],
        'back': [0],
        'wing': [8, 12],
        'tail': [13],
        'leg': [7, 11],
    }

    bird_w = bird_BB[2] - bird_BB[0]
    bird_h = bird_BB[3] - bird_BB[1]
    mask_w = int(bird_w / scale)
    mask_h = int(bird_h / scale)

    if resize_WH:
        mask_h = max(int(mask_w / 2), mask_h)
        mask_w = max(int(mask_h / 2), mask_w)

    img_IoU = {}
    for group_name, group_dims in groups.items():
        # print("group_name, group_dims:", group_name, group_dims)
        if group_name == 'others':
            continue

        img_IoU[group_name] = []
        # if the body part exist:
        # generate the iou for each attention map from each subgroup
        for group_dim in group_dims:
            for j, name in enumerate(names):
                mask = maps[name][idx, group_dim, :, :]
                mask = cv2.resize(mask, (224, 224))
                mask_BB_dict, (mask_c_x, mask_c_y) = generate_mask_BB(mask, bird_BB, scale, KNOW_BIRD_BB)

                KP_idxs = attri_to_group_dict[group_name]
                KPs_sub = [KPs[KP_idx][:2] for KP_idx in KP_idxs if KPs[KP_idx][2] != 0]
                if len(KPs_sub) == 0:
                    continue
                dis = [(point[0] - mask_c_x) ** 2 + (point[1] - mask_c_y) ** 2 for point in KPs_sub]
                gt_point = KPs_sub[np.argmin(dis)]
                KP_BB_dict = get_KP_BB(gt_point, mask_h, mask_w, bird_BB, KNOW_BIRD_BB)
                # print("KP_BB_dict", KP_BB_dict, mask_BB_dict)
                IoU = get_iou(KP_BB_dict, mask_BB_dict)
                img_IoU[group_name].append(IoU)
    return img_IoU


def generate_mask_BB(mask, bird_BB, scale=4, KNOW_BIRD_BB=False):
    """
    :return   a dict {'x1', 'x2', 'y1', 'y2'}
    """
    bird_w = bird_BB[2] - bird_BB[0]
    bird_h = bird_BB[3] - bird_BB[1]
    mask_w = int(bird_w / scale)
    mask_h = int(bird_h / scale)
    # np. np.max(mask)
    (mask_c_x, mask_c_y) = np.unravel_index(np.argmax(mask), np.array(mask).shape)
    mask_BB = get_KP_BB((mask_c_x, mask_c_y), mask_h, mask_w, bird_BB, KNOW_BIRD_BB)
    return mask_BB, (mask_c_x, mask_c_y)


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
    return {'x1': KP_x1, 'x2': KP_x2, 'y1': KP_y1, 'y2': KP_y2}



def read_KP(dir):
    KPs = np.load(dir + '.npy')
    bird_BB = np.load(dir + '_bird_BB.npy')
    # print("KPs.shape:", KPs.shape)
    # print("bird_BB.shape:", bird_BB.shape)
    return KPs, bird_BB


def tensor_imshow(inp):
    """Imshow for Tensor."""
    # inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = inp.detach().squeeze().cpu().numpy()
    # print("inp.shape:", inp.shape)
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(3):
        inp[:, i] = inp[:, i] * std[i] + mean[i]
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    # plt.imshow(inp, **kwargs)
    # if title is not None:
    #     plt.title(title)
    # print(inp.shape)
    return inp


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



def refine_attri_names(fn):
    attri_names = open(fn).readlines()
    assert len(attri_names) == 312
    for i in range(len(attri_names)):
        attri_names[i] = attri_names[i].strip().replace('::', '_').replace('(', '').replace(')', '')
    return attri_names





