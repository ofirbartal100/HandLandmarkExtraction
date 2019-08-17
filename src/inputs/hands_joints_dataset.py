import os
import re
import cv2
import numpy as np


def raw_dataset_to_dataframes(input_Images_folders_path, output_preprocessed_path):
    if not os.path.exists(output_preprocessed_path):
        os.makedirs(output_preprocessed_path)

    output_preprocessed_images_folder_path = output_preprocessed_path + "preprocessed_images/"
    if not os.path.exists(output_preprocessed_images_folder_path):
        os.makedirs(output_preprocessed_images_folder_path)

    dataframes_file = open(output_preprocessed_path + 'raw_dataframes.csv', 'w')
    dataframes_file.write(
        'image_name,Wx,Wy,T0x,T0y,T1x,T1y,T2x,T2y,T3x,T3y,I0x,I0y,I1x,I1y,I2x,I2y,I3x,I3y,M0x,M0y,M1x,M1y,M2x,M2y,M3x,M3y,R0x,R0y,R1x,R1y,R2x,R2y,R3x,R3y,L0x,L0y,L1x,L1y,L2x,L2y,L3x,L3y\n')

    media_extentions = ['.jpg', '.png', 'jpeg']

    count_folders = len(os.listdir(input_Images_folders_path))
    counter = 0
    for folder_name in os.listdir(input_Images_folders_path):
        counter += 1
        print("processing #{0} out of {1}".format(counter, count_folders))
        if not os.path.isdir(input_Images_folders_path + folder_name):
            continue
        current_folder_path = input_Images_folders_path + folder_name
        images_file_names = [fn for fn in os.listdir(current_folder_path)
                             if any(fn.endswith(ext) for ext in media_extentions)]
        for image_file_name in images_file_names:
            image_prefix = re.search("(\d{4})_", image_file_name).group(1)
            with open(
                    current_folder_path + "/" + image_prefix + "_joint2D.txt") as joints_file:  # Use file to refer to the file object
                data = joints_file.read()
                new_image_name = "{0}_{1}".format(folder_name, image_file_name)
                dataframe = "{0},{1}".format(new_image_name, data)
                dataframes_file.write(dataframe)

                # Load an color image in grayscale
                img = cv2.imread(current_folder_path + "/" + image_file_name, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(output_preprocessed_images_folder_path + new_image_name, img)


import matplotlib.pyplot as plt


def draw_skeleton(landmarks, style):
    # draw segments from wrist
    for i in [1, 5, 9, 13, 17]:
        plt.plot([landmarks[0, 0], landmarks[i, 0]], [landmarks[0, 1], landmarks[i, 1]], style)

    # draw fingers skeleton
    for f in [1, 5, 9, 13, 17]:
        for j in range(3):
            plt.plot([landmarks[f + j, 0], landmarks[f + j + 1, 0]], [landmarks[f + j, 1], landmarks[f + j + 1, 1]],
                     style)


def show_landmarks(image, true_lanndmarks, results_landmarks):
    """Show image with landmarks"""
    im = np.transpose(image.numpy(), (1, 2, 0))
    sq = im.squeeze()
    plt.imshow(sq, cmap='gray')
    draw_skeleton(true_lanndmarks, 'ro-')
    draw_skeleton(results_landmarks, 'bo-')


ganerates_dataset_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedDataset_extracted/GANeratedHands_Release/data/noObject/"
output_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/"
images_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + "preprocessed_images/"
csv_path_clean = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + 'raw_dataframes_clean.csv'
