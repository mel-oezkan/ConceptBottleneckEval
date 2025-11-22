"""
Evaluate trained models on the official CUB test set
"""
import math
import os
import random
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import CUBDatasetPartSegmentations
from CUB.config import BASE_DIR, N_ATTRIBUTES
from APN.apn_consts import PART_SEG_GROUPS
from APN.index_translation_util import map_attribute_ids_to_part_seg_group_id, get_attribute_names


def visualise(imgs, attention_masks, seg_masks, attribute_names, batch_nr, attributes=10, batch_idx=None, t_mean=0.5, t_std=2, save_path=""):
    """
        imgs: torch.Tensor of [B, C, H, W] the respective images
        attention_masks: torch.Tensor of [B, A, H, W] the attention masks, per attribute A
        seg_masks: torch.Tensor of [B, A, H, W] the segmentation masks, per attribute A
        attribute_names: List of [A], giving each attribute its name
        batch_nr: The current batch we are looking at
        attributes: Either a list of attributes that are to be visualized, or a number of how many random attributes to sample.
        batch_idx: The index in the batch from which we extract images, defaults to random one
        t_mean: The mean applied during preprocessing of the image
        t_std: The std applied during preprocessing of the image
        save_path: Path where to save the visualizations
    """

    B, A, H, W = attention_masks.shape
    # Sample random batches / attributes if none are provided
    if batch_idx is None:
        batch_idx = random.randint(0, B-1)
    if isinstance(attributes, int):
        attributes = random.sample(range(A), attributes)
    n_attributes = len(attributes)

    img = imgs[batch_idx]
    masks = attention_masks[batch_idx][attributes]
    seg_masks = seg_masks[batch_idx][attributes]
    attr_names = [attribute_names[i] for i in attributes]

    # Denorm image
    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    img_np = img_np * np.array(t_std) + np.array(t_mean)
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(2, n_attributes, figsize=(2*n_attributes, 5))

    for col in range(n_attributes):

        # Attribute name
        axes[0, col].set_title(attr_names[col], fontsize=10, pad=4)

        # Attention mask overlay
        attn_mask = masks[col].cpu().detach().numpy()
        axes[0, col].imshow(img_np)
        axes[0, col].imshow(attn_mask, cmap='jet', alpha=0.5)
        axes[0, col].axis('off')

        # Segmentation mask overlay
        seg_mask = seg_masks[col].cpu().detach().numpy()
        axes[1, col].imshow(img_np)
        axes[1, col].imshow(seg_mask, cmap='spring', alpha=0.5)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("ProtoNet Map")
    axes[1, 0].set_ylabel("Part Segmentation")

    # Create string for this img
    attr_str = "_".join([str(a) for a in attributes])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"b{batch_nr}_id{batch_idx}_attr{attr_str}.png"), dpi=200, bbox_inches='tight')
    plt.close()


def compute_soft_iou(att: torch.Tensor, gt: torch.Tensor, eps = 1e-7):
    """
        Computes the Intersection-over-Union between the attention masks and ground-truth
        segmentation masks, per attribute / part. Soft version, so the attention maps are
        raw and not binarized.

        Args:
            att: [B, A, H, W] in [0,1].
            gt : [B, A, H, W] in binary.
        Returns:
            soft_iou: The soft IoU score, per attribute and batch [B, A]
    """
    attf = att.float()
    gtf = gt.float()
    inter = (attf * gtf).sum(dim=(2,3))
    union = attf.sum(dim=(2,3)) + gtf.sum(dim=(2,3)) - inter
    return (inter + eps) / (union + eps)


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps = 1e-7):
    """
        Computes the Intersection-over-Union between the attention masks and ground-truth
        segmentation masks, per attribute / part. Hard version, so predicted masks must
        be binarized.

        Args:
            att: [B, A, H, W] in binary.
            gt : [B, A, H, W] in binary.
        Returns:
            iou: The IoU score, per attribute and batch [B, A]
    """
    pred = pred_mask.float()
    gt = gt_mask.float()
    inter = (pred * gt).sum(dim=(2,3))
    union = (pred + gt - pred*gt).sum(dim=(2,3))
    return (inter + eps) / (union + eps)


def percentile_threshold_maps(att: torch.Tensor, keep_ratio: float = 0.5):
    """
        Thresholds the attention maps to produce binary masks, with the threshold being
        computed dynamically per map to maintain keep_ratio% of the top activations for each map.

        Args:
            att: [B, A, H, W] in [0,1].
            keep_ratio: The ratio of top activations to keep, defaults to 0.5
        Returns:
            mask: Binarized attention masks [B, A, H, W].
    """
    B,A,H,W = att.shape
    N = H*W
    q = 1.0 - keep_ratio
    k = int(math.ceil(q * N))
    k = min(max(1, k), N)
    flat = att.reshape(B*A, N)
    kth_vals, _ = torch.kthvalue(flat, k, dim=1)   # k-th smallest
    thresh = kth_vals.view(B, A, 1, 1)
    mask = att > thresh
    return mask


def get_presence_mask(gt: torch.Tensor):
    """
        Given a ground truth tensor, this fct returns a boolean mask indicating the attribute
        presence for each [B, A] pair, as the ground truth segmentations for some images might
        exist and should thus be accounted for to not distort IoU computation.

        Args:
            gt: [B, A, H, W] binary ground truth segmentation mask per attribute
        Returns:
            mask: [B, A] boolean mask indicating concept presence
    """
    return gt.view(gt.size(0), gt.size(1), -1).any(dim=2)


def eval(args):
    """
        TODO
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(args.model_dir).to(device=device)

    if not hasattr(model, 'use_relu'):
        model.use_relu = True if args.use_relu else False

    if not hasattr(model, 'use_sigmoid'):
        model.use_sigmoid = True if args.use_sigmoid else False

    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None

    model.eval()

    data_dir = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

    transform_mean = 0.5
    transform_std = 2

    transform = transforms.Compose([
        transforms.CenterCrop(299), # Resolution
        transforms.ToTensor(), #implicitly divides by 255
        transforms.Normalize(
            mean = [transform_mean, transform_mean, transform_mean],
            std = [transform_std, transform_std, transform_std])
    ])

    dataset = CUBDatasetPartSegmentations(data_dir, args.use_attr, args.image_dir, args.part_seg_dir, 299, transform)

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,  
        pin_memory=True, 
        persistent_workers=True,
    )

    # This is our final tensor that maps each attribute by its id in the attention map to the respective part seg group
    map_attr_id_to_part_seg_group, unmatched_attr_id_mask = map_attribute_ids_to_part_seg_group_id(args.batch_size)
    map_attr_id_to_part_seg_group = map_attr_id_to_part_seg_group.to(device=device)
    unmatched_attr_id_mask = unmatched_attr_id_mask.to(device=device)

    # Get attribute names, remove unmatched attributes from it
    attribute_names = get_attribute_names("/".join(args.image_dir.split("/")[:-1]), used_attributes_only=True)
    unmatched_mask_list = unmatched_attr_id_mask.cpu().detach().tolist()
    attribute_names = [name for name, keep in zip(attribute_names, unmatched_mask_list) if keep]

    # Collecting IoU values across batches for proper mean
    A = len(attribute_names)
    soft_sum_per_attr = torch.zeros(A, device=device)
    soft_count_per_attr = torch.zeros(A, device=device)

    hard_sum_per_attr = torch.zeros(A, device=device)
    hard_count_per_attr = torch.zeros(A, device=device)

    with torch.no_grad():
        for data_idx, data in enumerate(loader):

            # Cast data to device
            data = [v.to(device) if torch.is_tensor(v) else v for v in data]

            inputs, labels, attr_labels, part_seg_masks = data
            attr_labels = torch.stack(attr_labels).t()  # N x A

            outputs, similarity_scores, attention_maps = model(inputs.to(device))
            
            # Normalize attention maps into [0, 1] range
            att_min = attention_maps.amin(dim=(2,3), keepdim=True)
            att_max = attention_maps.amax(dim=(2,3), keepdim=True)
            attention_maps = (attention_maps - att_min) / (att_max - att_min + 1e-7)

            # For each of the attributes attention map, get its respective group (and ID) and retrieve the segmentation mask for that group.
            _, _, H, W = part_seg_masks.shape
            group_idx = map_attr_id_to_part_seg_group.unsqueeze(0).expand(inputs.shape[0], -1)  # Adapt to current batch size [B, A]
            group_idx = group_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # Adapt to part segmentation size [B,A,H,W]

            # Gather segmentation mask for each attribute
            seg_masks_per_attribute = torch.gather(part_seg_masks, 1, group_idx)         # [B,A,H,W]

            # Mask out "other" or invalid attributes, so those for which we don't have a segmentation mask
            attention_maps = attention_maps[:, unmatched_attr_id_mask, :, :]

            # Upsample attention maps to segmentation mask shape, bilinear for smooth --> maybe choose other?
            attention_maps_upsampled = F.interpolate(attention_maps, size=(H, W), mode='bilinear', align_corners=False)

            # Compute soft IoU and IoU on binarized attention masks
            soft_iou_per_attr = compute_soft_iou(attention_maps_upsampled, seg_masks_per_attribute) # [B, A]
            binarized_att_maps = percentile_threshold_maps(attention_maps).float()
            binarized_att_maps_upsampled = F.interpolate(binarized_att_maps, size=(H, W), mode='nearest')
            hard_iou_per_attr = compute_iou(binarized_att_maps_upsampled, seg_masks_per_attribute)  # [B, A]

            # Get binary mask to denote for each image which attribute segmentations exist and which dont
            gt_presence_mask = get_presence_mask(seg_masks_per_attribute)   # [B, A]

            # Collect sum and count IoU for each attribute, mask out invalid entries
            soft_iou_per_attr[~gt_presence_mask] = 0 
            soft_sum_per_attr += soft_iou_per_attr.sum(dim=0)  # sum per attribute
            soft_count_per_attr += gt_presence_mask.sum(dim=0)  # count valid entries

            hard_iou_per_attr[~gt_presence_mask] = 0
            hard_sum_per_attr += hard_iou_per_attr.sum(dim=0)
            hard_count_per_attr += gt_presence_mask.sum(dim=0)

            # Visualise
            if data_idx % args.vis_every_n == 0:
                visualise(
                    inputs, attention_maps_upsampled, seg_masks_per_attribute, attribute_names,
                    data_idx, t_mean=transform_mean, t_std=transform_std, save_path=args.out_folder_path
                )
        
    # Compute global mIoU and attribute-wise mIoU
    miou_soft_per_attr = soft_sum_per_attr / (soft_count_per_attr + 1e-7)
    miou_hard_per_attr = hard_sum_per_attr / (hard_count_per_attr + 1e-7)
    miou_soft_global = soft_sum_per_attr.sum() / soft_count_per_attr.sum()
    miou_hard_global = hard_sum_per_attr.sum() / hard_count_per_attr.sum()

    print(f"Global soft mIoU: {miou_soft_global.item():.4f}")
    print(f"Global hard mIoU: {miou_hard_global.item():.4f}")

    print("\n--------- ATTRIBUTE-WISE mIoU ---------\n")
    for attr_id in range(miou_soft_per_attr.shape[0]):
        print(f"Attr {attribute_names[attr_id]} - soft mIoU: {miou_soft_per_attr[attr_id].item():.4f}, hard mIoU: {miou_hard_per_attr[attr_id].item():.4f}")

    print("\n--------- PARTSEG-WISE mIoU ---------\n")
    num_groups = map_attr_id_to_part_seg_group.max().item() + 1
    miou_soft_group_sum = torch.zeros(num_groups, device=miou_hard_per_attr.device)
    miou_soft_group_count = torch.zeros(num_groups, device=miou_hard_per_attr.device)
    miou_hard_group_sum = torch.zeros(num_groups, device=miou_hard_per_attr.device)
    miou_hard_group_count = torch.zeros(num_groups, device=miou_hard_per_attr.device)

    # Accumulate scores into groups
    for attr_id in range(len(miou_hard_per_attr)):
        g = map_attr_id_to_part_seg_group[attr_id].item()
        miou_soft_group_sum[g] += miou_soft_per_attr[attr_id]
        miou_soft_group_count[g] += 1
        miou_hard_group_sum[g] += miou_hard_per_attr[attr_id]
        miou_hard_group_count[g] += 1
    miou_soft_per_group = miou_soft_group_sum / (miou_soft_group_count + 1e-7)
    miou_hard_per_group = miou_hard_group_sum / (miou_hard_group_count + 1e-7)

    for group_id in range(miou_soft_per_group.shape[0]):
        print(f"Group {PART_SEG_GROUPS[group_id]} - soft mIoU: {miou_soft_per_group[group_id].item():.4f}, hard mIoU: {miou_hard_per_group[group_id].item():.4f}")



if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-part_seg_dir', default='', help='directory to the part segmentations')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')   
    parser.add_argument('-vis_every_n', type=int, default=15, help='Visualize a random example every vis_every_n batches')  
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 16

    # Create out folder
    out_folder_path = os.path.join(args.log_dir, "part_seg_vis")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_folder_path = out_folder_path

    # Create .txt for output of this script, write everything to there
    log_file = os.path.join(out_folder_path, "eval.txt")
    sys.stdout = open(log_file, "w")

    print(args)
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        result = eval(args)

