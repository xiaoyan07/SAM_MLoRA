"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import os
import cv2
import numpy as np
import random
from copy import deepcopy
import torchvision.transforms as f
from PIL import ImageFilter

def random_crop(image, crop_shape, mask=None):
    image_shape = image.shape
    image_shape = image_shape[0:2]
    ret = []
    nh = np.random.randint(0, image_shape[0] - crop_shape[0])
    nw = np.random.randint(0, image_shape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    ret.append(image_crop)
    if mask is not None:
        mask_crop = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        ret.append(mask_crop)
        return ret
    return ret[0]

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask

def blur(img, p=0.5):
    if np.random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def Ada_Hist(img):
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)
    clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    clahe_bgr_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

    return clahe_bgr_image

def default_loader(id, root, size = (512, 512)):
    img = cv2.imread(os.path.join(root, '{}.jpg').format(id))
    mask = cv2.imread(os.path.join(root + '{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    img, mask = random_crop(img, size, mask)

    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 1.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root, size):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.size = size

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root, self.size)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(list(self.ids))

def build_loader(id, root):
    img = cv2.imread(os.path.join(root, '{}.jpg').format(id))
    mask = cv2.imread(os.path.join(root + '{}.png').format(id), cv2.IMREAD_GRAYSCALE)

    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask

class build_ImageFolder(data.Dataset):
    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = build_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(list(self.ids))

def sp2build_loader(id, root, size=(640, 640)):
    image_path = os.path.join(root, '{}.png').format(id)
    img1 = cv2.imread(image_path)
    img = Ada_Hist(img1)
    mask_path = image_path.replace('RGB-PanSharpen', 'Mask')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img, mask = random_crop(img, size, mask)

    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask

class SP2Build_ImageFolder(data.Dataset):
    def __init__(self, trainlist, root, size):
        self.ids = trainlist
        self.loader = sp2build_loader
        self.root = root
        self.size = size

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root, self.size)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(list(self.ids))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8, 8))


if __name__ == '__main__':
    image = cv2.imread('G:/dataset_sum/9_SpaceNet2Build/AOI4/image/RGB-PanSharpen_AOI_4_Shanghai_img273.png')
    Ada_Hist(image)
