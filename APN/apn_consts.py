###                                                                                       ###
# NOTE: EVERY ID HERE IS ZERO-INDEXED; WHILE THE ATTRIBUTES IN THE CUB DATASET START FROM 1 #
###                                                                                       ###

# THIS MAPPING IS DEFINED BY APN
MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS = {
    "head": [278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 149, 150, 151, 0, 1, 2, 3, 4, 5, 6, 7, 8, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
    "breast": [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 54, 55, 56, 57, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    "belly": [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 244, 245, 246, 247],
    "back": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 236, 237, 238, 239, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    "wing": [308, 309, 310, 311, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 212, 213, 214, 215, 216],
    "tail": [73, 74, 75, 76, 77, 78, 240, 241, 242, 243, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181],
    "leg": [263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277],
    "others": [248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 217, 218, 219, 220, 221]
}

# THIS MAPPING IS FROM CUB ATTRIBUTE IDS->CBM; AS CBM ONLY USES 112 ATTRIBUTES
CBM_SELECTED_CUB_ATTRIBUTE_IDS = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]


# GROUPS AS DEFINED BY THE PART SEGMENTATION DATASET FOR CUB, IMPORTANT: left + right masks are joined to one
PART_SEG_GROUPS = ['body', 'head', 'neck', 'beak', 'tail', 'wing', 'leg', 'eye']


# HOW THESE ARE ASSEMBLED:
#   - right / left eye: eye ring + all eye color attributes
#   - neck: all nape colors
#   - beak: all bill shapes, all bill colors, all bill lengths
#   - body: take all attributes from groups back, breast and belly
#   - head: take all attributes from group head, but remove all from: eyes, neck, beak
#   - the rest is just an equal mapping, no distinction into left / right.
# TODO: Attributes with ID 220, 221, 225, 249, 253, 254, 259, 260, 262, 235 could not be matched, but they are about main shape or color
# TODO: So this could be associated to body, or to all areas
MAP_PART_SEG_GROUPS_TO_CUB_ATTRIBUTE_IDS = {
    'eye': [100, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
    'neck': [182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196],
    'beak': [0, 1, 2, 3, 4, 5, 6, 7, 8, 149, 150, 151, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292], 
    'body': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['back'] + MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['breast'] + MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['belly'],
    'head': [293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104],
    'tail': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['tail'],
    'wing': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['wing'],
    'leg': MAP_APN_GROUPS_TO_CUB_ATTRIBUTE_IDS['leg'],
}

# Maps the groups as defined by the part segmentations (collapsing left / right to one) to the part groups as defined by CUB itself in parts/parts.txt
MAP_PART_SEG_GROUPS_TO_CUB_GROUPS = {
    'eye': ["left eye", "right eye"],
    'neck': ["nape"],
    'beak': ["beak"],
    'body': ["back", "belly", "breast", "throat"],
    'head': ["crown", "forehead"],
    'tail': ["tail"],
    'wing': ["left wing", "right wing"],
    'leg': ["left leg", "right leg"],
}


"""

OLD

# THIS MAPPING MAPS FROM CBM-SELECTED ATTRIBUTES TO THE GROUPS AS DEFINED BY APN, CAUTION: THESE ARE RELATIVE INDICES
# indeces are based on the selected attributes only - generated in index_translation_util.py
CUB_SELECTED_ATTRIBUTES_PER_GROUP = {
    'head': [99, 100, 101, 50, 102, 103, 104, 105, 106, 107, 53, 54, 55, 56, 57, 58, 51, 52, 0, 1, 2, 3, 64, 65, 66, 67, 68, 69, 37, 38],
    'breast': [45, 46, 47, 48, 49, 22, 23, 24, 39, 40, 41, 42, 43, 44],
    'belly': [16, 17, 18, 19, 20, 21, 70, 71, 72, 73, 74, 75, 89],
    'back': [10, 11, 12, 13, 14, 15, 83, 84, 85, 25, 26, 27, 28, 29, 30],
    'wing': [108, 109, 110, 111, 4, 5, 6, 7, 8, 9, 76, 77],
    'tail': [31, 86, 87, 88, 32, 33, 34, 35, 36, 59, 60, 61, 62, 63],
    'leg': [96, 97, 98], 
    'others': [90, 91, 92, 93, 94, 95, 81, 82, 78, 79, 80]
}

# GROUPS AS DEFINED BY APN
CUB_GROUPS = ['head', 'belly', 'breast', 'back', 'wing', 'tail', 'leg', 'others']
"""

NUM_ATTRIBUTES = 112