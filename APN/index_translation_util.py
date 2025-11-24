import torch
import os

from APN.apn_consts import MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS, CBM_SELECTED_CUB_ATTRIBUTE_IDS, MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS, PART_SEG_GROUPS


# Build a mapping from old index -> new index
def map_attribute_ids_from_cub_to_cbm(absolute_indices: list):
    return [relative_index for relative_index, absolute_index in enumerate(CBM_SELECTED_CUB_ATTRIBUTE_IDS) if absolute_index in absolute_indices]


def map_attribute_ids_to_part_seg_group_id(batch_size, verbose=True):
    """
        Creates a lookup tensor that, given the index of an attribute within our model output, maps it to its original attribute
        ID, maps it to the selected ones, and then maps it to the ID of its corresponding part segmentation group.
    """

    # Build a lookup dict from the attributes IDs to the part seg group ID {attribute_index: group_id}
    attr_to_group = {}
    for group_id, group_name in enumerate(PART_SEG_GROUPS):
        for idx in MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS[group_name]:
            if verbose and idx in attr_to_group:
                print(f"Attribute ID {idx} is assigned to multiple groups: Old group = {attr_to_group[idx]}, New group = {group_id}")
            attr_to_group[idx] = group_id

    # Create lookup tensor from it: For an attribute idx, map to its actual attribute ID, and use it for the lookup
    lookup = torch.empty(len(CBM_SELECTED_CUB_ATTRIBUTE_IDS), dtype=torch.long)
    unmatched_indices = []
    for attr_idx in range(len(CBM_SELECTED_CUB_ATTRIBUTE_IDS)):

        # This happens because of the "other" group, fill with dummy value, remove in main script from attention map!
        if CBM_SELECTED_CUB_ATTRIBUTE_IDS[attr_idx] not in attr_to_group.keys():
            if verbose:
                print(f"Attribute with ID {CBM_SELECTED_CUB_ATTRIBUTE_IDS[attr_idx]} could not be matched to any part segmentation group.")
            unmatched_indices.append(attr_idx)
            continue

        lookup[attr_idx] = attr_to_group[CBM_SELECTED_CUB_ATTRIBUTE_IDS[attr_idx]]

    # Remove unmatched attributes: Create mask stating which entries should be thrown out
    mask = torch.ones(lookup.size(0), dtype=torch.bool)
    mask[unmatched_indices] = False
    lookup_clean = lookup[mask]

    # This is our final tensor that maps each attribute by its id in the attention map to the respective part seg group
    return lookup_clean, mask


def get_attribute_names(path_to_cub_data, used_attributes_only=True):
    attribute_names = []
    with open(os.path.join(path_to_cub_data, 'attributes', 'attributes.txt'), "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip this attribute if unused and the flag is set
            if used_attributes_only and idx not in CBM_SELECTED_CUB_ATTRIBUTE_IDS:
                continue

            _, attr_name = line.split(" ", 1)
            attribute_names.append(attr_name)

    return attribute_names

