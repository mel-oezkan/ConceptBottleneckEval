"""
Evaluate trained models on the official CUB test set
"""
import os
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


def visualise(imgs, attention_masks, seg_masks, attribute_names, n_attributes=10, batch_idx=7, save_path=""):
    """
        imgs: torch.Tensor of [B, C, H, W] the respective images
        attention_masks: torch.Tensor of [B, A, H, W] the attention masks, per attribute A
        seg_masks: torch.Tensor of [B, A, H, W] the segmentation masks, per attribute A
        attribute_names: List of [A], giving each attribute its name
        batch_idx: The batch from which we extract images
        n_attributes: How many of the first n attributes to visualize
    """
    img = imgs[batch_idx, :, :, :]
    masks = attention_masks[batch_idx, :n_attributes, :, :]
    seg_masks = seg_masks[batch_idx, :n_attributes, :, :]
    attr_names = attribute_names[:n_attributes]

    img_np = img.permute(1, 2, 0).cpu().numpy()  # H x W x C
    fig, axes = plt.subplots(2, n_attributes, figsize=(2*n_attributes, 5))

    # TODO: DENORM IMAGE

    for col in range(n_attributes):

        # Attribute name
        # axes[0, col].text(0.5, 0.5, attr_names[col], ha='center', va='center', fontsize=10)
        # axes[0, col].axis('off')
        # Title above attention image (row index 1)
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
    # fig.text(0.01, 0.66, "ProtoNet Map", va='center', rotation='vertical', fontsize=12)
    # fig.text(0.01, 0.36, "Part Segmentation", va='center', rotation='vertical', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "attribute_part_seg_vis_b7.png"), dpi=200, bbox_inches='tight')


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

    transform = transforms.Compose([
        transforms.CenterCrop(299), # Resolution
        transforms.ToTensor(), #implicitly divides by 255
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
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

    for data_idx, data in enumerate(loader):

        # Cast data to device
        data = [v.to(device) if torch.is_tensor(v) else v for v in data]

        inputs, labels, attr_labels, part_seg_masks = data
        attr_labels = torch.stack(attr_labels).t()  # N x A

        inputs_var = torch.autograd.Variable(inputs).cuda()

        outputs, similarity_scores, attention_maps = model(inputs_var)

        # For each of the attributes attention map, get its respective group (and ID) and retrieve the segmentation mask for that group.
        _, _, H, W = part_seg_masks.shape
        group_idx = map_attr_id_to_part_seg_group.unsqueeze(-1).unsqueeze(-1)   # [B,A,1,1]
        group_idx = group_idx.expand(-1, -1, H, W)                              # [B,A,H,W]

        # Gather segmentation mask for each attribute
        seg_masks_per_attribute = torch.gather(part_seg_masks, 1, group_idx)         # [B,A,H,W]

        # Mask out "other" or invalid attributes, so those for which we don't have a segmentation mask
        attention_maps = attention_maps[:, unmatched_attr_id_mask, :, :]

        # Upsample attention maps to segmentation mask shape, bilinear for smooth --> maybe choose other?
        attention_maps_upsampled = F.interpolate(attention_maps, size=(H, W), mode='bilinear', align_corners=False)

        # Visualise
        visualise(inputs, attention_maps_upsampled, seg_masks_per_attribute, attribute_names, save_path=args.log_dir)

        assert 0 == 1

        # # TODO TODO HANDLE EMPTY PART SEG MASKS AS THEY ARE JUST PLACEHOLDERS TO ENABLE BATCHING TODO TODO # #
        # --> calculate as long as possible with placeholder but mask out before computing the score!
        
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
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 16

    print(args)
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        result = eval(args)

