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
        self.dir = os.path.join(opt.dataroot, 'test')

        fp = open(opt.data_list)
        s = fp.read()
        self.videos = json.loads(s)

        self.frame = self.videos["frame"]
        self.video_name = self.videos["name"]
        for i in range(1, len(self.frame)):
            self.frame[i] = self.frame[i] + self.frame[i - 1]
        self.i = 0

        self.images_dir = '/media/data2/zhangpz/ai_dancing/results/ai_dance_wgan_sn/test_latest/images/'
        self.input_dir = os.path.join(opt.dataroot, 'test')
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
        print("ok", index, self.i)
        index = index
        while index + 1 > self.frame[self.i]:
            self.i = self.i + 1
        to_video_name = self.video_name[self.i]
        image_i = 0
        if self.i == 0:
            image_i = index
        else:
            image_i = index - self.frame[self.i - 1]
        to_image_name = '{:04}.jpg'.format(image_i)

        input_p_path = to_video_name + '/' + to_image_name

        input_PB_path = to_video_name + '/poses/' + to_image_name

        input_BG_path = to_video_name + '/background/' + "0000.jpg"

        #input_GT_path = to_video_name + '/images/' + to_image_name

        #mask_person_path = to_video_name + '/mask/' + to_image_name + '.npy'

        #mask_bg_path = to_video_name + '/mask/' + to_image_name

        #input_seg_person_path = to_video_name + '/segmentation/' + to_image_name

        input_P = Image.open(os.path.join(self.images_dir, input_p_path)).convert('RGB')
        input_PB = Image.open(os.path.join(self.dir, input_PB_path)).convert('RGB')
        input_BG = Image.open(os.path.join(self.dir, input_BG_path)).convert('RGB')


        input_P_t = self.transform(input_P)
        input_PB_t = self.transform(input_PB)

        input_BG_t = self.transform(input_BG)


        return {'P': input_P_t, 'PB': input_PB_t, 'BG': input_BG_t, 'GT': input_BG_t,
                'MP': input_BG_t, "MBG": input_BG_t, "SP": input_BG_t, "SBG": input_BG_t,
                'P_path': to_video_name + '0000.jpg',
                'BG_path': to_video_name + '{:04}.jpg'.format(image_i)}

    def __len__(self):
        if self.opt.phase == 'train':
            return self.size
        elif self.opt.phase == 'test':
            return 50000

    def name(self):
        return 'KeyDataset'
