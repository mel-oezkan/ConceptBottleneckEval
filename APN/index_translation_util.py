import torch
import os

from APN.apn_consts import CUB_ATTRIBUTES_PER_GROUP, CUB_SELECTED_ATTRIBUTES, CUB_GROUPS_TO_PART_SEG, PART_SEG_GROUPS


# Build a mapping from old index -> new index
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(CUB_SELECTED_ATTRIBUTES)}

# Translate each group's indices into new ones (keeping only those present in the subset)
CUB_SELECTED_ATTRIBUTES_PER_GROUP = {
    group: [old_to_new[i] for i in indices if i in old_to_new]
    for group, indices in CUB_ATTRIBUTES_PER_GROUP.items()
}

# print(CUB_SELECTED_ATTRIBUTES_PER_GROUP)


def map_attribute_ids_to_part_seg_group_id(batch_size):
    """
        Creates a lookup tensor that, given the index of an attribute within our model output, maps it to its original attribute
        ID, maps it to the selected ones, and then maps it to the ID of its corresponding part segmentation group.
    """

    # Build a lookup dict from the attributes IDs to the part seg group ID {attribute_index: group_id}
    attr_to_group = {}
    for group_id, group_name in enumerate(PART_SEG_GROUPS):
        for idx in CUB_GROUPS_TO_PART_SEG[group_name]:
            if idx in attr_to_group:
                print(f"Attribute ID {idx} is assigned to multiple groups: Old group = {attr_to_group[idx]}, New group = {group_id}")
            attr_to_group[idx] = group_id

    # Create lookup tensor from it: For an attribute idx, map to its actual attribute ID, and use it for the lookuü
    lookup = torch.empty(len(CUB_SELECTED_ATTRIBUTES), dtype=torch.long)
    unmatched_indices = []
    for attr_idx in range(len(CUB_SELECTED_ATTRIBUTES)):

        # This happens because of the "other" group, fill with dummy value, remove in main script from attention map!
        if CUB_SELECTED_ATTRIBUTES[attr_idx] not in attr_to_group.keys():
            print(f"Attribute with ID {CUB_SELECTED_ATTRIBUTES[attr_idx]} could not be matched to any part segmentation group.")
            unmatched_indices.append(attr_idx)
            continue

        lookup[attr_idx] = attr_to_group[CUB_SELECTED_ATTRIBUTES[attr_idx]]

    # Remove unmatched attributes: Create mask stating which entries should be thrown out
    mask = torch.ones(lookup.size(0), dtype=torch.bool)
    mask[unmatched_indices] = False
    lookup_clean = lookup[mask]

    # Batch it to enable efficient lookup
    # This is our final tensor that maps each attribute by its id in the attention map to the respective part seg group
    return lookup_clean.unsqueeze(0).expand(batch_size, -1), mask


def get_attribute_names(path_to_cub_data, used_attributes_only=True):
    attribute_names = []
    with open(os.path.join(path_to_cub_data, 'attributes', 'attributes.txt'), "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip this attribute if unused and the flag is set
            if used_attributes_only and idx not in CUB_SELECTED_ATTRIBUTES:
                continue

            _, attr_name = line.split(" ", 1)
            attribute_names.append(attr_name)

    return attribute_names
