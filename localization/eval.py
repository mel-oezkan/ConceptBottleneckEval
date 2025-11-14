#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
import numpy as np
import torch

from utils import *
from statistics import mean


def test_with_IoU(opt, model, testloader, vis_groups=None, vis_root=None,
                  required_num=2, save_att=False, sub_group_dic=None, group_dic=None):
    """
    save feature maps to visualize the activation
    :param model: loaded model
    :param testloader:
    #:param attribute: test attributes (model input, no effect to activation maps here)
    #:param test_classes: test classes (accuracy input, no effect to activation maps here)
    :param vis_groups: the groups to be shown
    :param vis_layer: the layers to be shown
    :param vis_root: save path to activation maps
    :param required_num: the number images shown in each categories
    :return:
    """
    # print('Calculating the IoU of attention maps, saving attention map to:', save_att)
    layer_name = 'layer4'
    #GT_targets = []
    #predicted_labels = []
    vis_groups = group_dic

    whole_IoU = []
    # print("vis_groups:", vis_groups)
    # required_num = 2  # requirement denotes the number for each categories
    with torch.no_grad():
        count = dict()
        for i, (input, target, impath) in \
                enumerate(testloader):
            save_att_idx = []
            labels = target.data.tolist()
            for idx in range(len(labels)):
                label = labels[idx]
                if label in count:
                    count[label] = count[label] + 1
                else:
                    count[label] = 1
                if count[label] <= required_num:
                    save_att_idx.append(1)
                else:
                    save_att_idx.append(0)

            if opt.cuda:
                input = input.cuda()
                target = target.cuda()

            output, pre_attri, attention, _ = model(input)#, attribute)
            # pre_attri.shape : 64， 312
            # attention.shape : 64, 312, 7, 7
            maps = {layer_name: attention['layer4'].cpu().numpy()}
            pre_attri = pre_attri['layer4']
            target_groups = [{} for _ in range(output.size(0))]  # calculate the target groups for each image
            # target_groups is a list of size image_num
            # each item is a dict, including the attention index for each subgroup
            for part in vis_groups.keys():
                sub_group = sub_group_dic[part]
                keys = list(sub_group.keys())

                # sub_activate_id is the attention id for each part in each image. The size is img_num * sub_group_num
                sub_activate_id = []
                for k in keys:
                    sub_activate_id.append(torch.argmax(pre_attri[:, sub_group[k]], dim=1, keepdim=True))
                sub_activate_id = torch.cat(sub_activate_id, dim=1).cpu().tolist()  # (batch_size, sub_group_dim)
                for attention_id, argdims in enumerate(sub_activate_id):
                    target_groups[attention_id][part] = [sub_group[keys[i]][argdim] for i, argdim in enumerate(argdims)]

            KP_root = './data/vis/save_KPs/'
            scale = opt.IoU_scale
            batch_IoU = calculate_atten_IoU(input, impath, save_att_idx, maps, [layer_name], target_groups, KP_root,
                                            save_att=save_att, scale=scale, resize_WH=opt.resize_WH,
                                            KNOW_BIRD_BB=opt.KNOW_BIRD_BB)
            # print('batch_IoU:', batch_IoU)
            whole_IoU += batch_IoU
            #_, predicted_label = torch.max(output.data, 1)
            #predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            #GT_targets = GT_targets + target.data.tolist()
            # break

    body_avg_IoU, mean_IoU = calculate_average_IoU(whole_IoU, IoU_thr=opt.IoU_thr)
    #GT_targets = np.asarray(GT_targets)
    #acc_all, acc_avg = compute_per_class_acc(map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
    #                                 np.array(predicted_labels), test_classes.numpy())
    return body_avg_IoU, mean_IoU



def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label


def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = np.array(test_label)
    predicted_label = np.array(predicted_label)
    acc_per_class = []
    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in range(len(nclass)):
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class)/len(acc_per_class)



def calculate_average_IoU(whole_IoU, IoU_thr=0.5):
    img_num = len(whole_IoU)
    body_parts = whole_IoU[0].keys()
    body_avg_IoU = {}
    for body_part in body_parts:
        body_avg_IoU[body_part] = []
        body_IoU = []
        for im_id in range(img_num):
            if len(whole_IoU[im_id][body_part]) > 0:
                if_one = []
                for item in whole_IoU[im_id][body_part]:
                    if_one.append(1 if item > IoU_thr else 0)
                body_IoU.append(mean(if_one))
        body_avg_IoU[body_part].append(mean(body_IoU))
    num = 0
    sum = 0
    for part in body_avg_IoU:
        if part != 'tail':
            sum += body_avg_IoU[part][0]
            num += 1
    # print(sum/num *100)
    return body_avg_IoU, sum/num *100





