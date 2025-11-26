#test run of loc accuracy
print("starting script")
import argparse
from localization.eval import test_CUB_IoU
from CUB.dataset import CUBDataset

import torch
from torchvision import transforms
print("imports done")

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

CBM_SELECTED_CUB_ATTRIBUTE_IDS = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

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


def map_attribute_ids_from_cub_to_cbm(absolute_indices: list):
    return [relative_index for relative_index, absolute_index in enumerate(CBM_SELECTED_CUB_ATTRIBUTE_IDS) if absolute_index in absolute_indices]


MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS = {part: map_attribute_ids_from_cub_to_cbm(attr_ids) for part, attr_ids in MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS.items()}

#print(MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS)

def parse_args():
    parser = argparse.ArgumentParser(description='Test localization accuracy on CUB dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the processed CUB data (class_attr_data_10)')
    parser.add_argument('--cub_root', type=str, required=True, help='Path to CUB_200_2011 dataset root')
    parser.add_argument('--resol', type=int, default=299, help='Image resolution')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Starting script")
    print("Setting up transforms")
    
    resol = args.resol
    transform = transforms.Compose([
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])
    
    print("Loading dataset")
    pkl_paths = [f"{args.data_root}/test.pkl"]
    cub_test_dataset = CUBDataset(
        pkl_paths, 
        use_attr=False, 
        uncertain_label=False,
        no_img=False,
        image_dir="images",
        n_class_attr=2,
        transform=transform
    )

    print("Loading model")
    model = torch.load(args.model_path)

    print("Calling eval function")
    body_avg_IoU, mean_IoU = test_CUB_IoU(
        args={
            "cuda": torch.cuda.is_available(),
            "IoU_thr": args.iou_thr
        }, 
        model=model, 
        dataset=cub_test_dataset,
        CUB_root=args.cub_root,
        part_attribute_mapping=MAP_CUB_PARTS_GROUPS_TO_CUB_ATTRIBUTE_IDS,
        subgroup_mapping=MAP_PART_SEG_GROUPS_TO_CUB_GROUPS
    )

    print(f"Body Average IoU: {body_avg_IoU}")
    print(f"Mean IoU: {mean_IoU}")


if __name__ == "__main__":
    main()