import cv2
from PIL import Image
from torchvision import transforms

from .input_handler import *


class PILGrayImageInputHandler(InputHandler):

    def _handle(self, *args):
        """
        abstract method override to handle VideoFrame input.
        :param args: path to video , *shape of output frame
        :return: PILGray image
        """
        path = args[0]
        pil_gray = Image.open(path).convert('L')

        try:
            shape = args[1]
            if shape:
                pil_gray = pil_gray.resize(shape)
        except:
            pass
        return pil_gray


class TensorImageInputHandler(InputHandler):

    def __init__(self, mean=0.485, std=0.229):
        self.mean = mean
        self.std = std

    def _handle(self, *args):
        """
        abstract method override to handle VideoFrame input.
        :param args: path to video , *shape of output frame
        :return: Tensor
        """
        pil_handler = PILGrayImageInputHandler()
        pil_image = pil_handler.handle(*args)
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,),
                                 (self.std,))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        tensor = in_transform(pil_image)[:, :, :]
        return tensor
