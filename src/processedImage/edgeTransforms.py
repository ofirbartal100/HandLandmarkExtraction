import numpy as np
import torch
import cv2
import abc

from src.processedImage import ProcessedImage


class ProcessedImageTransform(abc.ABC):
    def __init__(self, show_landmarks_flag):
        self._show_landmarks_flag = show_landmarks_flag

    @abc.abstractmethod
    def _transform(self, processedImage):
        pass

    def transform(self, processedImage):
        transformed_img = self._transform(processedImage)
        if self._show_landmarks_flag:
            return self._show_landmarks(transformed_img, processedImage.landmarks)
        return transformed_img

    def _show_landmarks(self, img, landmarks):
        img_clone = img.copy()
        for source_x, source_y, target_x, target_y in ProcessedImage.get_fingers_joints(landmarks):
            cv2.line(img_clone, (source_x, source_y), (target_x, target_y), (255, 0, 0), 2)
        return img_clone


class CannyTransform(ProcessedImageTransform):
    def __init__(self, l_hesteresys, h_hesteresys, show_landmarks_flag=False):
        super().__init__(show_landmarks_flag)
        self.low_hesteresys = l_hesteresys
        self.high_hesteresys = h_hesteresys

    def _transform(self, processedImage):
        return cv2.Canny(processedImage.img, self.low_hesteresys, self.high_hesteresys)


class IdentityTransform(ProcessedImageTransform):
    def __init__(self, show_landmarks_flag=False):
        super().__init__(show_landmarks_flag)

    def _transform(self, processedImage):
        return processedImage.img.copy()


class SteeredEdgeTransform(ProcessedImageTransform):
    def __init__(self, k_size, mask_size, show_landmarks_flag=False):
        super().__init__(show_landmarks_flag)
        self._mask_size = mask_size
        self._k_size = k_size

    def _transform(self, processedImage):
        transformed_img = np.zeros(processedImage.img.shape[:2], np.uint8)
        for source_x, source_y, target_x, target_y in ProcessedImage.get_fingers_joints(processedImage.landmarks):
            hist_mask = Mask(processedImage.img.shape[:2], [source_x, source_y], self._mask_size)

            kernel_down, kernel_up = self.create_kernel(
                torch.Tensor([source_x, source_y]).double(),
                torch.Tensor([target_x, target_y]).double(),
                self._k_size)

            filtered_down = cv2.filter2D(processedImage.img, cv2.CV_8U, kernel_down)
            filtered_up = cv2.filter2D(processedImage.img, cv2.CV_8U, kernel_up)

            filtered_down = hist_mask(filtered_down)
            filtered_up = hist_mask(filtered_up)

            transformed_img = cv2.bitwise_or(transformed_img, filtered_down)
            transformed_img = cv2.bitwise_or(transformed_img, filtered_up)

        return transformed_img

    def create_kernel(self, source, target, kernel_size):
        direction = target - source
        # flip since image is in flipped coordinates
        direction *= torch.from_numpy(np.array([1.0, -1.0]))
        direction /= direction.norm()

        if kernel_size == 5:
            sobely = SteeredEdgeTransform.sobely_5
        else:
            sobely = SteeredEdgeTransform.sobely_3
        sobelx = sobely.t()

        direction_filter = sobelx * direction[0] + sobely * direction[1]
        return direction_filter.numpy(), direction_filter.numpy() * -1

    # 5x5 sobel y filter
    sobely_5 = torch.from_numpy(np.array([
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [4., 2., 0., -2., -4.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
    ]))

    # 3x3 sobel y filter
    sobely_3 = torch.from_numpy(np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]
    ]))

class Mask(object):
    def __init__(self, shape, point, mask_size):
        self._mask = np.zeros(shape, np.uint8)
        x_range_start, x_range_stop, y_range_start, y_range_stop = Mask.extract_mask_location(point, mask_size)
        self._mask[y_range_start:y_range_stop, x_range_start:x_range_stop] = 255

    @staticmethod
    def extract_mask_location(point, mask_size):
        return int(point[0] - mask_size / 2), int(point[0] + mask_size / 2), int(point[1] - mask_size / 2), int(
            point[1] + mask_size / 2)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            return self._mask
        try:
            return cv2.bitwise_and(*args, *args, mask=self._mask)
        except:
            raise
