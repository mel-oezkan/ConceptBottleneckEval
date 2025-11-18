#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
import numpy as np
import torch
import torch.nn.functional as F

from utils import *
from statistics import mean
import os
from CUB.dataset import CUBDataset

from utils import get_KP_BB, get_iou


#rewrite of method because this sucks
def test_CUB_IoU(args:dict, model, dataset:CUBDataset, CUB_root:str):
    #reimplementation of paper description
    
    #save ious from images
    whole_IoU = []

    #get relevant paths to mappings
    imgID_imgName_mapping_path, parts_locs_path, parts_mapping_path, bird_BB_path = get_CUB_paths(CUB_root)

    #read here later into dictionary
    imgName_to_imgID = {}
    with open(imgID_imgName_mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_str, *text_parts = line.split()
            key = " ".join(text_parts)
            imgName_to_imgID[key] = int(id_str)

    img_id_to_part_locs = {}
    with open(parts_locs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_str, *info = line.split()
            
            info = [int(x) for x in info]

            if int(id_str) not in img_id_to_part_locs:
                img_id_to_part_locs[int(id_str)] = [info]
            else:
                img_id_to_part_locs[int(id_str)] = img_id_to_part_locs[int(id_str)] + [info]

    imgID_to_birdBB = {}
    with open(bird_BB_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_str, *bb_info = line.split()

            bb_info = [int(x) for x in bb_info]
            
            imgID_to_birdBB[int(id_str)] = bb_info



    with torch.no_grad():
        for i, (input, target, impath) in enumerate(dataset):
            #for each image, first find relevant data
            img_shape = input.shape
            #get image id
            img_name = os.sep.join(impath.split(os.sep)[-2:]) #class path is also in mapping name
            img_id = imgName_to_imgID[img_name]
            #get location and visibility infos of parts of image
            part_infos = img_id_to_part_locs[img_id]
            #get bird bb of image
            bird_bb = imgID_to_birdBB[img_id] #fun fact readme says it is x, y, width, height, but APN later wants x1, y1, x2, y2

            bounding_boxes_per_part = get_BB_per_part(bird_bb, part_infos)

            if args.cuda:
                input = input.cuda()
            
            output, pre_attri, attention, _ = model(input)


            #get argmax attribute per part
            argmax_per_part = None #max index per part, -1 if part not present
            #take attention if given else empty
            heatmaps = [attention[idx] if idx != -1 else [] for idx in argmax_per_part]

            #upscale heatmaps, use cv2 like original code
            heatmap = [cv2.resize(m, img_shape[-2:]) for m in heatmaps]

            optimal_masks = [get_optimal_mask(heatmap, part_box) for heatmap, part_box in zip(heatmaps, bounding_boxes_per_part)]

            IoUs = [get_iou({"x1":op_mask[0], "y1":op_mask[1], "x2":op_mask[2], "y2":op_mask[3]}, {"x1":gt_mask[0], "y1":gt_mask[1], "x2":gt_mask[2], "y2":gt_mask[3]}) \
                    if op_mask != [] else -1 for op_mask, gt_mask in zip(optimal_masks, bounding_boxes_per_part)]

           
            whole_IoU += IoUs

    body_avg_IoU, mean_IoU = calculate_average_IoU(whole_IoU, IoU_thr=args.IoU_thr)
    return body_avg_IoU, mean_IoU


def get_optimal_mask(heatmap, part_bb):
    #takes the heatmap and
    if heatmap == [] or part_bb == []: #normally both should be empty but just in case make or
        return []
    
    mask_w = part_bb[2] - part_bb[0]
    mask_h = part_bb[3] - part_bb[1]

    kernel = torch.ones((1, 1, mask_h, mask_w))
    response = F.conv2d(heatmap, kernel)

    max_pos = response.argmax()

    best_y = (max_pos // response.shape[-1]).item()
    best_x = (max_pos %  response.shape[-1]).item()

    #x1, y1, x2, y2
    return [best_x - int(mask_w/2), best_y - int(mask_h/2), best_x + int(mask_w/2), best_y + int(mask_h/2)]

def get_BB_per_part(bird_bb, part_infos, scale=4):
    #generate a bounding box mask per part, empty if part is not visible
    #BB format returned is (x1, y1, x2, y2)

    width = bird_bb[2]
    height = bird_bb[3]

    mask_w = width / scale
    mask_h = height / scale

    part_masks = []
    for info in part_infos:
        #info: part_id, x1, y1, visible
        if info[-1] == 0: #part not visible, no gt
            part_masks.append([])
            continue
        gt = (info[1], info[2])
        #change from x, y, w, h to x1, y1, x2, y2
        transformed_bird_BB = [bird_bb[0], bird_bb[1], bird_bb[0] + bird_bb[2], bird_bb[1] + bird_bb[3]]
        part_masks.append(get_KP_BB(gt, mask_h, mask_w, transformed_bird_BB))

    return part_masks


def get_CUB_paths(CUB_root:str):
    #get path to certain files starting from CUB root, assuming CUB structure

    #maps image id to image name
    imgID_imgName_mapping_path = os.path.join(CUB_root, "images.txt")
    #information about parts bounding boxes and in-image occurence
    parts_locs_path = os.path.join(CUB_root, "parts", "parts_loc.txt") 
    #maps part ID to part name
    parts_mapping_path = os.path.join(CUB_root, "parts", "parts.txt")
    #information about bird bounding boxes per image id
    bird_BB_path = os.path.join(CUB_root, "bounding_boxes.txt")

    return imgID_imgName_mapping_path, parts_locs_path, parts_mapping_path, bird_BB_path





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
            # pre_attri.shape : 64，112
            # attention.shape : 64, 112, 8, 8
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





