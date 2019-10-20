import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
import math

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt, i_j_w_h=(1,2,3,4)):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'enhance':
        transform_list.append(transforms.Lambda(
            lambda img: __random_crop(img, i_j_w_h)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __random_crop(img, i_j_w_h):
    i, j, h, w = i_j_w_h
    if 0 < w <= img.size[0] and 0 < h <= img.size[1]:
        img = img.crop((j, i, j + w, i + h))
    else:
        frame = Image.new('RGB', (w, h), (0, 0, 0))
        frame.paste(img, (j, i))
        img = frame
    img = img.resize((272, 480), Image.BICUBIC)
    return img

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def __scale_width2(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.NEAREST)

def get_specific_transform1(opt):
    transform_list = []
    transform_list.append(transforms.Lambda(
        lambda img: __scale_width2(img, opt.fineSize)))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def get_specific_transform2(opt):
    transform_list = [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_specific_transform3(opt):
    transform_list = []
    transform_list.append(transforms.Lambda(
        lambda img: __scale_width2(img, opt.fineSize)))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)




