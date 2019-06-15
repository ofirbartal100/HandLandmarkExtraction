from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import torch
import torch.optim as optim
import requests
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, models
import torchvision.transforms.functional as TTF


# helper function for loading in any type and size of image.
# The load_image function also converts images to normalized Tensors
def load_image_RGB(img_path, max_size=None, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    size = image.size
    if max_size is not None:

        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# helper function for loading in any type and size of image.
# The load_image function also converts images to normalized Tensors
def load_image_GRAY(img_path, max_size=None, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('L')
    else:
        image = Image.open(img_path).convert('L')
    size = image.size
    if max_size is not None:

        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,),
                             (0.229,))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:, :, :]

    return image


def load_image_GRAY_crop(img_path, min_x, min_y, max_x, max_y):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('L')
    else:
        image = Image.open(img_path).convert('L')

    cropped_image = TTF.crop(image, min_x, min_y, max_x - min_x, max_y - min_y)
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,),
                             (0.229,))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(cropped_image)[:, :, :]

    return image


# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert_fromRGB(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert_fromGRAY(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image * np.array((0.229,)) + np.array((0.485,))
    image = image.clip(0, 1)

    return image


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    # plt.imshow(image,cmap='gray')
    im = np.transpose(image.numpy(), (1, 2, 0))
    sq = im.squeeze()
    pltimage.imsave("imsave.png", sq, cmap='gray')
    plt.imshow(sq, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.savefig('savefig.jpg', bbox_inches='tight')  # pause a bit so that plots are updated



