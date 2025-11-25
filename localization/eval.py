#code from the APN github: https://github.com/wenjiaXu/APN-ZSL/blob/master/model/main_utils.py#L295
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from statistics import mean
import os
from CUB.dataset import CUBDataset

from localization.utils import get_KP_BB, get_iou



MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS = {
    'left eye': [100, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
    'right eye': [100, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
    'nape': [182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196],
    'back': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['back'],
    'belly': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['belly'],
    'beak': [0, 1, 2, 3, 4, 5, 6, 7, 8, 149, 150, 151, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292], 
    'breast': [54, 55, 56, 57, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'throat': [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104],
    'crown': [293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307],
    'forehead': [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166],
    'tail': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['tail'],
    'left wing': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['wing'],
    'right wing': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['wing'],
    'left leg': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['leg'],
    'right leg': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['leg'],
}


#TO-DO: CUB parts to Sebbis seg groups mapping einbauen
def test_CUB_IoU(args:dict, model, dataset:CUBDataset, CUB_root:str, part_attribute_mapping:dict, subgroup_mapping:dict):
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
            line = line.strip()
            id, part_name = line.split(" ", maxsplit=1)
            part_dict[int(id)] = part_name
            

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
            
            info = [float(x) for x in info]

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

            bb_info = [float(x) for x in bb_info]
            
            imgID_to_birdBB[int(id_str)] = bb_info

    #start eval

    with torch.no_grad():
        for i, (input, target, impath) in enumerate(dataset):
            if i % 10 == 0:
                print(f"Processing image {i}/{len(dataset)}")

            #for each image, first find relevant data
            #image size to resize heatmaps
            im_width, im_height = input.size()[-2:]
            #get image id
            img_name = os.sep.join(impath.split(os.sep)[-2:]) #class path is also in mapping name
            img_id = imgName_to_imgID[img_name]
            #get location and visibility infos of parts of image
            part_infos = img_id_to_part_locs[img_id]
            #get bird bb of image
            bird_bb = imgID_to_birdBB[img_id] #fun fact readme says it is x, y, width, height, but APN later wants x1, y1, x2, y2
            #for each part get the gt bounding box
            bounding_boxes_per_part = get_BB_per_part(bird_bb, part_infos)

            if args["cuda"]:
                input = input.to("cuda")

            input = input.unsqueeze(0)  #add batch dimension
           
            _, pre_attri, attention = model(input)

            pre_attri = pre_attri.squeeze(0).cpu().numpy()  #remove batch dimension and move to cpu
            attention = attention.squeeze(0).cpu().numpy()  #move to cpu
   
            #get argmax attribute per part
            argmax_per_part = [] #max index per part, -1 if part not present
            
            for part in part_dict.values():
                #take argmax of each part group
                arg_max = max(part_attribute_mapping[part], key=lambda i: pre_attri[i])
                argmax_per_part.append(arg_max)
       
            #take heatmap if given else empty
            heatmaps = [attention[idx] if idx != -1 else np.ndarray([]) for idx in argmax_per_part]

            #upscale heatmaps, use cv2 like original code
            heatmaps = [cv2.resize(m, dsize=(im_width, im_height)) for m in heatmaps]
         
            #compute sliding window from heatmaps
            optimal_masks = [get_optimal_mask(heatmap, part_box) for heatmap, part_box in zip(heatmaps, bounding_boxes_per_part)]
            #get iou per existing part, empty if none
            IoUs = [get_iou({"x1":op_mask[0], "y1":op_mask[1], "x2":op_mask[2], "y2":op_mask[3]}, {"x1":gt_mask[0], "y1":gt_mask[1], "x2":gt_mask[2], "y2":gt_mask[3]}) \
                    if op_mask != [] else -1 for op_mask, gt_mask in zip(optimal_masks, bounding_boxes_per_part)]
           
            #take computed IoUs per part and store them to dict for later calculation
            sub_res = {}
            for part, iou in zip(list(part_dict.values()), IoUs):
                sub_res[part] = iou
           
            whole_IoU.append(sub_res)
            
    #calculate avarages
    body_avg_IoU, mean_IoU = calculate_average_partwise_acc(whole_IoU, subgroup_mapping, IoU_thr=args["IoU_thr"])
    return body_avg_IoU, mean_IoU


def calculate_average_partwise_acc(all_ious:list[dict], subgroup_mapping:dict, IoU_thr: float=0.5):
    #compute acc with ious and threshold for all images per part
    #all ious = list with each item having a matching from part to iou

    #preprocessing, merge groups that belong together and take the max iou value
    processed_ious = []
    for iou in all_ious:
        new_part = {}
        for part in subgroup_mapping.keys():
            grouped_parts = subgroup_mapping[part]
            max_iou = -1
            for g_part in grouped_parts:
                if g_part in iou.keys():
                    if iou[g_part] > max_iou:
                        max_iou = iou[g_part]
            new_part[part] = max_iou
        processed_ious.append(new_part)

    #collect part ious over all images
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

    

def get_optimal_mask(heatmap:np.ndarray, part_bb:list):
    #takes the heatmap and
    
    if heatmap.size == 0 or len(part_bb) == 0: #normally both should be empty but just in case make or
        return []
    
    #mask height and width
    mask_w = int(part_bb[2] - part_bb[0])
    mask_h = int(part_bb[3] - part_bb[1])

    #conv for response map, take maximum value
    kernel = torch.ones((1, 1, mask_h, mask_w))
    response = F.conv2d(torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    
    #argmax from flatten, then convert back to 2d pos
    flat_response = torch.flatten(response)
    max_pos = flat_response.argmax()
    best_y = (max_pos // heatmap.shape[-1]).item()
    best_x = (max_pos %  heatmap.shape[-1]).item()

    #x1, y1, x2, y2
    return [max(0, best_x - int(mask_w/2)), max(0, best_y - int(mask_h/2)), min(heatmap.shape[0], best_x + int(mask_w/2)), min(heatmap.shape[1], best_y + int(mask_h/2))]

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
        if info[-1] == 0.0: #part not visible, no gt
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
    parts_locs_path = os.path.join(CUB_root, "parts", "part_locs.txt") 
    #maps part ID to part name
    parts_mapping_path = os.path.join(CUB_root, "parts", "parts.txt")
    #information about bird bounding boxes per image id
    bird_BB_path = os.path.join(CUB_root, "bounding_boxes.txt")

    return imgID_imgName_mapping_path, parts_locs_path, parts_mapping_path, bird_BB_path




