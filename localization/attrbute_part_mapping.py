#mapping of attributes for each original part in the CUB parts/parts.txt list

PARTS_LIST = ["back", "beak", "belly", "breast", "breast", "forehead",
              "left eye", "left leg", "left wing", "nape", "right eye",
              "right leg", "right wing", "tail", "throat"]

PARTS_DICT = {
    0: "back", 
    1: "beak", 
    2: "belly", 
    3: "breast", 
    4: "breast", 
    5: "forehead",
    6: "left eye", 
    7: "left leg", 
    8: "left wing", 
    9: "nape", 
    10: "right eye", 
    11: "right leg", 
    12: "right wing", 
    13: "tail", 
    14: "throat"
}

#indices are attribute indices, starting by 1
# contains all attributes from the CUB_SELECTED_ATTRIBUTES list in apn_consts
# relies on the CUB_SELECTED_ATTRIBUTES_PER_GROUP mapping for inspo
#attirbutes like wing color is written for both left and right wing
"""ATTRIBUTE_PART_MAPPING = {
    "back": [25, 29, 30, 35, 36, 38, 236, 238, 239, 59, 63, 64, 69, 70, 72], #assumptions: has_upperparts_color means back
    "beak": [1, 4, 6, 7], 
    "belly": [40, 44, 45, 50, 51, 53, 198, 202, 203, 208, 209, 211, 244], 
    "breast", 
    "breast", 
    "forehead",
    "left eye", 
    "left leg", 
    "left wing": [10, 14, 15, 20, 21, 23, ], 
    "nape", 
    "right eye", 
    "right leg", 
    "right wing": [10, 14, 15, 20, 21, 23, ], 
    "tail", 
    "throat"
}"""

CUB_SELECTED_ATTRIBUTES = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]


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

print(CUB_SELECTED_ATTRIBUTES[89])

for b in CUB_SELECTED_ATTRIBUTES_PER_GROUP["belly"]:
    print(CUB_SELECTED_ATTRIBUTES[b])