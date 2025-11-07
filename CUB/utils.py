"""
Common functions for visualization in different ipython notebooks
"""
import os
import random
from matplotlib.pyplot import figure, imshow, axis, show
from matplotlib.image import imread

N_CLASSES = 200
N_ATTRIBUTES = 312

def get_class_attribute_names(img_dir = 'CUB_200_2011/images/', feature_file='CUB_200_2011/attributes/attributes.txt'):
    """
    Returns:
    class_to_folder: map class id (0 to 199) to the path to the corresponding image folder (containing actual class names)
    attr_id_to_name: map attribute id (0 to 311) to actual attribute name read from feature_file argument
    """
    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split('.')[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)

    attr_id_to_name = dict()
    with open(feature_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ')
            attr_id_to_name[int(idx) - 1] = name
    return class_to_folder, attr_id_to_name

def sample_files(class_label, class_to_folder, number_of_files=10):
    """
    Given a class id, extract the path to the corresponding image folder and sample number_of_files randomly from that folder
    """
    folder = class_to_folder[class_label]
    class_files = random.sample(os.listdir(folder), number_of_files)
    class_files = [os.path.join(folder, f) for f in class_files]
    return class_files

def show_img_horizontally(list_of_files):
    """
    Given a list of files, display them horizontally in the notebook output
    """
    fig = figure(figsize=(40,40))
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image)
        axis('off')
    show(block=True)


def get_cub_attributes() -> list[int]:
    """helper function to get the list of all CUB attributes"""
    return [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]