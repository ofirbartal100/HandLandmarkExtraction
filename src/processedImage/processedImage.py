import numpy as np
import torch
import cv2
from pathlib import Path


class ProcessedImage(object):
    #     CR: doc? why do you have a class without functionality?
    def __init__(self, row):
        self.row = int(row)

        self._images_path = r"D:\TAU\Research\Project\Dataset\GANeratedDataset_v3\GANeratedHands_Release\data\noObject\\"
        self._csv_path = r"D:\TAU\Research\Project\Learning\Results\LandmarksExtraction\raw_dataframes_clean_fromt_hand.csv"

        # load and extract data
        with open(self._csv_path, 'r') as csv:
            line = csv.readlines()[self.row]
            line_values = line.split(',')
        image_name = line_values[0].split('_', 1)
        full_image_path = Path(self._images_path).joinpath(image_name[0] + '\\').joinpath(image_name[1])
        landmarks = np.array(line_values[1:], dtype=float)

        self.landmarks = torch.from_numpy(landmarks).view(-1, 2)
        self.img = cv2.imread(str(full_image_path), cv2.IMREAD_GRAYSCALE)

    # represents the base joints position of every finger
    BaseJoints = [1, 5, 9, 13, 17]
    # represents the amount of joints in every finger
    BaseJointLength = 3

    @staticmethod
    def get_fingers_joints(landmarks):
        for i in ProcessedImage.BaseJoints:
            for j in range(ProcessedImage.BaseJointLength):
                source_x = int(landmarks[i + j][0].item())
                source_y = int(landmarks[i + j][1].item())
                target_x = int(landmarks[i + j + 1][0].item())
                target_y = int(landmarks[i + j + 1][1].item())
                yield source_x, source_y, target_x, target_y
