from apn_consts import CUB_ATTRIBUTES_PER_GROUP, CUB_SELECTED_ATTRIBUTES

# Build a mapping from old index -> new index
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(CUB_SELECTED_ATTRIBUTES)}

# Translate each group's indices into new ones (keeping only those present in the subset)
CUB_SELECTED_ATTRIBUTES_PER_GROUP = {
    group: [old_to_new[i] for i in indices if i in old_to_new]
    for group, indices in CUB_ATTRIBUTES_PER_GROUP.items()
}

print(CUB_SELECTED_ATTRIBUTES_PER_GROUP)