import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
import json
import cv2
import math


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        fp = open(opt.data_list)
        s = fp.read()
        self.videos = json.loads(s)
        self.size = len(self.videos['name'])
        self.dir = os.path.join(opt.dataroot, 'train')
        self.use_seg = opt.use_seg
        self.use_inverse_seg = opt.use_inverse_seg
        # self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        # self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.use_enhance = False
        self.use_fashion = False
        self.use_test = False
        if opt.phase == 'train':
            self.use_test = opt.use_test
            self.use_fashion = opt.use_fashion
            if opt.use_test:
                fp_test = open('/media/data2/zhangpz/ai_dancing/datasets/final_test_videos.json')
                s_test = fp_test.read()
                self.videos_test = json.loads(s_test)
                self.size_test = len(self.videos_test['name'])
            if opt.use_fashion:
                fashion_name = os.listdir('/media/data2/zhangpz/ai_dancing/datasets/ai_dance/fashion')
                self.fashion = fashion_name
                self.size_fashion = len(self.fashion)

            # self.init_categories(opt.pairLst)
            self.use_enhance = 0
            if opt.resize_or_crop == 'enhance':
                self.use_enhance = 1


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
        if self.use_enhance == 1:
            i_j_w_h_1 = self.get_i_j_h_w(scale=(0.6, 1.4), ratio=(0.5666666, 0.5666667))
            i_j_w_h_2 = self.get_i_j_h_w(scale=(0.6, 1.4), ratio=(0.5666666, 0.5666667))
            self.transform1 = get_transform(self.opt, i_j_w_h_1)
            self.transform2 = get_transform(self.opt, i_j_w_h_2)
        else:
            self.transform = get_transform(self.opt)
        if self.use_test:
            k = random.randint(1,1000)
            if k == 500:
                self.dir = os.path.join(self.opt.dataroot, 'test')
                index = random.randint(0, self.size_test - 1)
                video_name = self.videos_test['name'][index]
                #print(video_name)
                frame_from = 0
                frame_to = 0

            elif self.use_fashion and 490 <= k <= 510:
                # if self.opt.phase == 'train':
                self.dir = os.path.join(self.opt.dataroot, 'fashion')
                index = random.randint(0, self.size_fashion - 1)
                video_name = self.fashion[index]
                #print(video_name)
                frame_from = 0
                frame_to = 0
            else:
            #if self.opt.phase == 'train':
                self.dir = os.path.join(self.opt.dataroot, 'train')
                index = random.randint(0, self.size - 1)

                video_name = self.videos['name'][index]
                frame_from = random.randint(0, self.videos['frame'][index]-1)
                frame_to = random.randint(0, self.videos['frame'][index]-1)
        else:
            self.dir = os.path.join(self.opt.dataroot, 'train')
            index = random.randint(0, self.size - 1)

            video_name = self.videos['name'][index]
            frame_from = random.randint(0, self.videos['frame'][index] - 1)
            frame_to = random.randint(0, self.videos['frame'][index] - 1)
        '''
        while frame_to == frame_from:
            frame_to = random.randint(0, self.videos['frame'][index]-1)
        '''
        if self.use_inverse_seg:
            MP1_name = video_name + '/mask/' + '{:04}.jpg.npy'.format(frame_from)
            MP2_name = video_name + '/mask/' + '{:04}.jpg.npy'.format(frame_to)
        if self.use_seg or self.use_inverse_seg:
            P1_name = video_name + '/segmentation/' + '{:04}.jpg'.format(frame_from)
            P2_name = video_name + '/segmentation/' + '{:04}.jpg'.format(frame_to)
        else:
            P1_name = video_name + '/images/' + '{:04}.jpg'.format(frame_from)
            P2_name = video_name + '/images/' + '{:04}.jpg'.format(frame_to)


        BP1_name = video_name + '/poses/' + '{:04}.jpg'.format(frame_from)
        BP2_name = video_name + '/poses/' + '{:04}.jpg'.format(frame_to)

        P1_path = os.path.join(self.dir, P1_name)  # person 1
        BP1_path = os.path.join(self.dir, BP1_name)  # bone of person 1
        if self.use_inverse_seg:
            MP1_path = os.path.join(self.dir, MP1_name)  # mask of person 1


        P2_path = os.path.join(self.dir, P2_name)  # person 2
        BP2_path = os.path.join(self.dir, BP2_name)  # bone of person 2
        if self.use_inverse_seg:
            MP2_path = os.path.join(self.dir, MP2_name)  # mask of person 2

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = Image.open(BP1_path).convert('RGB')
        BP2_img = Image.open(BP2_path).convert('RGB')

        if self.use_inverse_seg:
            MP1_img = (np.load(MP1_path)-0.5)*2
            MP2_img = (np.load(MP2_path)-0.5)*2
        '''
        w, h = P1_img.size
        if h == 1280:
            P1_img = P1_img.resize((544, 960), Image.ANTIALIAS)
            P2_img = P2_img.resize((544, 960), Image.ANTIALIAS)
            BP1_img = BP1_img.resize((544, 960), Image.ANTIALIAS)
            BP2_img = BP2_img.resize((544, 960), Image.ANTIALIAS)
            h = 960
        if h != 960:
            #print(h)
            shift = int((960 - h)/2)
            P1_img_t = Image.new('RGB', (w, 960), (0, 0, 0))
            P1_img_t.paste(P1_img, (0, shift, w, shift + h))

            P2_img_t = Image.new('RGB', (w, 960), (0, 0, 0))
            P2_img_t.paste(P2_img, (0, shift, w, shift + h))

            BP1_img_t = Image.new('RGB', (w, 960), (0, 0, 0))
            BP1_img_t.paste(BP1_img, (0, shift, w, shift + h))

            BP2_img_t = Image.new('RGB', (w, 960), (0, 0, 0))
            BP2_img_t.paste(BP2_img, (0, shift, w, shift + h))

            P1_img = P1_img_t
            P2_img = P2_img_t
            BP1_img = BP1_img_t
            BP2_img = BP2_img_t
        '''
        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0, 1)

            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = BP1_img.transpose(Image.FLIP_LEFT_RIGHT)
                BP2_img = BP2_img.transpose(Image.FLIP_LEFT_RIGHT)

            BP1 = self.transform(BP1_img)
            BP2 = self.transform(BP2_img)
            if self.use_inverse_seg:
                MP1 = self.transform(MP1_img)
                MP2 = self.transform(MP2_img)

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            if self.use_enhance == 1:
                BP1 = self.transform1(BP1_img)
                BP2 = self.transform2(BP2_img)

                P1 = self.transform1(P1_img)
                P2 = self.transform2(P2_img)
            else:
                BP1 = self.transform(BP1_img)
                BP2 = self.transform(BP2_img)

                P1 = self.transform(P1_img)
                P2 = self.transform(P2_img)

            if self.use_inverse_seg:
                if self.opt.resize_or_crop == 'scale_width':
                    oh, ow = MP1_img.shape
                    if ow != self.opt.fineSize:
                        w = self.opt.fineSize
                        h = int(self.opt.fineSize * oh / ow)
                        MP1 = torch.from_numpy(np.array([cv2.resize(MP1_img, (w, h), interpolation=cv2.INTER_NEAREST)])).float()
                        MP2 = torch.from_numpy(np.array([cv2.resize(MP2_img, (w, h), interpolation=cv2.INTER_NEAREST)])).float()
        if self.use_inverse_seg:
            P1 = torch.cat((P1, MP1), 0)
            P2 = torch.cat((P2, MP2), 0)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': video_name + '{:04}.jpg'.format(frame_from),
                'P2_path': video_name + '{:04}.jpg'.format(frame_to)}

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return 45000

    def name(self):
        return 'KeyDataset'

    def get_i_j_h_w(self, scale, ratio):
        img = [544,960]
        area = img[0] * img[1]

        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        i = random.randint(0, abs(img[1] - h))
        j = random.randint(0, abs(img[0] - w))
        return i, j, h, w