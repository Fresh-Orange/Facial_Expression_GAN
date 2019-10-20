import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_specific_transform1, get_specific_transform2, get_specific_transform3
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
import json
import cv2

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, 'train')
        self.images = os.listdir('/media/data2/zhangpz/ai_dancing/datasets/blend_image/')
        self.size = len(self.images)
        # self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)
        self.transform1 = get_specific_transform1(opt)
        self.transform2 = get_specific_transform2(opt)
        self.transform3 = get_specific_transform3(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        input_p_path = self.images[index]
        to_video_name = input_p_path[0:5]
        to_image_name = input_p_path[7:15]
        to_bg_name = input_p_path[17:25]

        input_PB_path = to_video_name + '/poses/' + to_image_name


        input_BG_path = to_video_name + '/background/' + to_bg_name

        input_GT_path = to_video_name + '/images/' + to_image_name

        mask_person_path = to_video_name + '/mask/' + to_image_name + '.npy'

        mask_bg_path = to_video_name + '/mask/' + to_bg_name + '.npy'

        input_seg_person_path = to_video_name + '/segmentation/' + to_image_name

        input_P = Image.open(os.path.join('/media/data2/zhangpz/ai_dancing/datasets/blend_image/', input_p_path)).convert('RGB')
        input_PB = Image.open(os.path.join(self.dir, input_PB_path)).convert('RGB')
        input_BG = Image.open(os.path.join(self.dir, input_BG_path)).convert('RGB')
        input_GT = Image.open(os.path.join(self.dir, input_GT_path)).convert('RGB')
        mask_person = np.load(os.path.join(self.dir, mask_person_path))
        mask_bg = np.load(os.path.join(self.dir, mask_bg_path))
        input_seg_person = Image.open(os.path.join(self.dir, input_seg_person_path)).convert('RGB')

        input_P_t = self.transform(input_P)
        input_PB_t = self.transform(input_PB)

        input_BG_t1 = self.transform1(input_BG)
        input_GT_t = self.transform(input_GT)

        input_seg_person_t = self.transform3(input_seg_person)

        if self.opt.resize_or_crop == 'scale_width':
            oh, ow = mask_person.shape
            if ow != self.opt.fineSize:
                w = self.opt.fineSize
                h = int(self.opt.fineSize * oh / ow)
                mask_person_t = torch.from_numpy(
                    np.array([cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)])).float()
                mask_bg_t = torch.from_numpy(
                    np.array([cv2.resize(mask_bg, (w, h), interpolation=cv2.INTER_NEAREST)])).float()
        oness = torch.ones(mask_bg_t.shape)
        mask_bg_t = torch.mul((oness - mask_bg_t), (oness - mask_person_t))
        input_seg_bg_t1 = torch.mul(input_BG_t1, torch.cat((mask_bg_t, mask_bg_t, mask_bg_t), 0))

        input_BG_t = self.transform2(input_BG_t1)
        input_seg_bg_t = self.transform2(input_seg_bg_t1)


        return {'P': input_P_t, 'PB': input_PB_t, 'BG': input_BG_t, 'GT': input_GT_t,
                'MP': mask_person_t, "MBG": mask_bg_t, "SP": input_seg_person_t, "SBG": input_seg_bg_t,
                'P_path': self.images[index],
                'BG_path': to_video_name + to_bg_name}

    def __len__(self):
        if self.opt.phase == 'train':
            return 45000
        elif self.opt.phase == 'test':
            return 45000

    def name(self):
        return 'KeyDataset'
