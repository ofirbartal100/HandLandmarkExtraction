import cv2
from PIL import Image
from torchvision import transforms

from .input_handler import *


class PILGrayVideoFrameInputHandler(InputHandler):

    def _handle(self, *args):
        """
        abstract method override to handle VideoFrame input.
        :param args: path to video , frame_num , *shape of output frame
        :return: PILGray image
        """
        path = args[0]
        frame_num = args[1]
        if args[2] is not None:
            shape = args[2]

        cap = cv2.VideoCapture(path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if 0 <= frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if shape:
                frame = cv2.resize(frame, shape)
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            pil_gray = pil_im.convert('L')
            return pil_gray
        return None


class TensorVideoFrameInputHandler(InputHandler):

    def __init__(self, mean=0.485, std=0.229):
        self.mean = mean
        self.std = std

    def _handle(self, *args):
        """
        abstract method override to handle VideoFrame input.
        :param args: path to video , frame_num , *shape of output frame
        :return: Tensor
        """
        pil_handler = PILGrayVideoFrameInputHandler()
        pil_image = pil_handler.handle(*args)
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,),
                                 (self.std,))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        tensor = in_transform(pil_image)[:, :, :]
        return tensor
