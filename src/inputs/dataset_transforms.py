import torch
import numpy as np
import torchvision.transforms.functional as TTF
import torch.nn
import torch.optim
from abc import ABC, abstractmethod


class DatasetTransform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self):
        pass


class Rescale(DatasetTransform):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        super().__init__()
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def transnform(self, img, label):
        """
        transform the image and label together
        :param img: PIL grayscale image
        :param label: 21 landmarks
        :return: PIL grayscale image, 21 landmarks
        """
        w, h = img.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        transformed_img = TTF.resize(img, self.output_size)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        transformed_label = label * [new_w / w, new_h / h]
        return transformed_img, transformed_label


class CropToBoundry(DatasetTransform):
    def transnform(self, img, label):
        """
        transform the image and label together
        :param img: PIL grayscale image
        :param label: 21 landmarks
        :return: PIL grayscale image, 21 landmarks
        """
        w, h = img.size
        max_xy = np.max(label, 0) + 10  # padding
        min_xy = np.min(label, 0) - 10  # padding

        max_xy = np.clip(max_xy, 0, [w, h])
        min_xy = np.clip(min_xy, 0, [w, h])
        new_wh = max_xy - min_xy
        cropped_image = TTF.crop(img, min_xy[1], min_xy[0], new_wh[1], new_wh[0])

        cropped_joints_pos = label - min_xy
        return cropped_image, cropped_joints_pos


class Compose(DatasetTransform):
    def __init__(self, transforms):
        """
        :param transforms: iterable of DatasetTransform
        """
        super().__init__()
        self.transforms = transforms

    def transnform(self, img, label):
        """
        apply transforms to the image and label together
        :param img: PIL grayscale image
        :param label: 21 landmarks
        :return: PIL grayscale image, 21 landmarks
        """
        for transform in self.transforms:
            img, label = transform.transform(img, label)

        return img, label


class ToTensor(DatasetTransform):

    def transnform(self, img, label):
        """
        apply transforms to the image and label together
        :param img: PIL grayscale image
        :param label: 21 landmarks
        :return: tensor for image, tensor for landmarks
        """
        return TTF.to_tensor(img), torch.from_numpy(label).type(torch.FloatTensor).view(-1)


class Normalize(DatasetTransform):
    def __init__(self, mean, std):
        """
        can only be applied after ToTensor, since using the tensor transform of the image and label
        :param mean: double for mean
        :param std: double for std
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def transnform(self, img_tensor, label_tensor):
        """
        apply transforms to the image and label together
        :param img: tensor transform of the PIL grayscale image
        :param label: tensor transform of the 21 landmarks
        :return: normalize image tensor by mean and std , same label
        """
        return TTF.normalize(img_tensor, self.mean, self.std), label_tensor
