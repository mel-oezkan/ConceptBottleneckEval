#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
import numpy as np
import torch
import torch.nn.functional as F

from utils import *
from statistics import mean
import os
from CUB.dataset import CUBDataset

from utils import get_KP_BB, get_iou


#TO-DO: CUB parts to Sebbis seg groups mapping einbauen
def test_CUB_IoU(args:dict, model, dataset:CUBDataset, CUB_root:str, part_attribute_mapping:dict):
    # part_attribute_mapping means a dict that maps from a part (CUB/parts/parts.txt) to a list of attribute IDs
    # sum of all attribute IDs must not be more that number of attention maps returned/attributes used by model. also is
    # required to be 0 indexed and correctly match the attribute order in the model

    #reimplementation of paper description
    
    #save ious from all images
    whole_IoU = []

    #get relevant paths to mappings
    imgID_imgName_mapping_path, parts_locs_path, parts_mapping_path, bird_BB_path = get_CUB_paths(CUB_root)

    part_dict = {}
    with open(parts_mapping_path) as ps:
        for line in ps:
            id, part_name = line.split(" ")
            part_dict[id] = part_name
            

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
            
            _, pre_attri, attention, _ = model(input)


            #get argmax attribute per part
            argmax_per_part = [] #max index per part, -1 if part not present
            
            for part in part_dict.values():
                #take argmax of each part group
                arg_max = max(part_attribute_mapping[part], key=lambda i: pre_attri[i])
                argmax_per_part.append(arg_max)

            #take attention if given else empty
            heatmaps = [attention[idx] if idx != -1 else [] for idx in argmax_per_part]

            #upscale heatmaps, use cv2 like original code
            heatmaps = [cv2.resize(m, img_shape[-2:]) for m in heatmaps]

            optimal_masks = [get_optimal_mask(heatmap, part_box) for heatmap, part_box in zip(heatmaps, bounding_boxes_per_part)]

            IoUs = [get_iou({"x1":op_mask[0], "y1":op_mask[1], "x2":op_mask[2], "y2":op_mask[3]}, {"x1":gt_mask[0], "y1":gt_mask[1], "x2":gt_mask[2], "y2":gt_mask[3]}) \
                    if op_mask != [] else -1 for op_mask, gt_mask in zip(optimal_masks, bounding_boxes_per_part)]

            #take computed IoUs per part and store them to dict for later calculation
            sub_res = {}
            for part, iou in zip(list(part_dict.values()), IoUs):
                sub_res[part] = iou
           
            whole_IoU += IoUs

    body_avg_IoU, mean_IoU = calculate_average_partwise_acc(whole_IoU, IoU_thr=args.IoU_thr)
    return body_avg_IoU, mean_IoU

#import part mapping
#TO-DO: change code so that it merges all parts with the same keys for a given part mapping
def calculate_average_partwise_acc(all_ious:list[dict], IoU_thr: float=0.5):
    #compute acc with ious and threshold for all images per part
    #all ious = list with each item having a matching from part to iou


    #for things we have 2 of (eyes, wings and leg) we take the max value
    #preprocessing
    processed_ious = []
    for iou in all_ious:
        new_part = {}
        for part in iou.keys():
            if "leg" in part or "eye" in part or "wing" in part:
                continue
            new_part[part] = iou[part]

        new_part["legs"] = max(iou["left leg"], iou["right leg"])
        new_part["legs"] = max(iou["left eye"], iou["right eye"])
        new_part["legs"] = max(iou["left wing"], iou["right wing"])

    collect = {}
    for part in processed_ious[0].keys():
        collect[part] = []

    for iou in processed_ious:
        for part, value in iou.items():
            if value == -1: #this part was not present in the image, no iou
                continue
            collect[part].append(1 if value >= IoU_thr else 0)

    #divide sum by amount
    res = {}
    for part, collected_ious in collect.items():
        res[part] = sum(collected_ious)/len(collected_ious) if len(collected_ious) != 0 else -1

    mean_iou = [x for x in res.values() if x != -1]
    mean_iou_acc = sum(mean_iou)/len(mean_iou)

    return res, mean_iou_acc

    


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




