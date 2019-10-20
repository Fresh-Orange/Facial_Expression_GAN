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

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        fp = open(opt.data_list)
        s = fp.read()
        self.videos = json.loads(s)

        self.frame = self.videos["frame"]
        self.video_name = self.videos["name"]
        for i in range(1, len(self.frame)):
            self.frame[i] = self.frame[i] + self.frame[i - 1]
        self.i = 0

        self.size = len(self.videos['name'])
        self.dir = os.path.join(opt.dataroot, 'test')
        self.use_seg = opt.use_seg
        self.use_inverse_seg = opt.use_inverse_seg
        # self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        # self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints

        # self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

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
        #if self.opt.phase == 'train':
        if index + 1 > self.frame[self.i]:
            self.i = self.i + 1
        video_name = self.video_name[self.i]
        if self.i == 0:
            image_i = index
        else:
            image_i = index - self.frame[self.i - 1]
        if self.use_inverse_seg:
            MP1_name = video_name + '/mask/' + '0000.jpg.npy'
        if self.use_seg or self.use_inverse_seg:
            P1_name = video_name + '/segmentation/' + '0000.jpg'
        else:
            P1_name = video_name + '/images/' + '0000.jpg'

        #print(index % 300)
        BP1_name = video_name + '/poses/' + '0000.jpg'
        BP2_name = video_name + '/poses/' + '{:04}.jpg'.format(image_i)

        P1_path = os.path.join(self.dir, P1_name)  # person 1
        BP1_path = os.path.join(self.dir, BP1_name)  # bone of person 1
        if self.use_inverse_seg:
            MP1_path = os.path.join(self.dir, MP1_name)  # mask of person 1

        BP2_path = os.path.join(self.dir, BP2_name)  # bone of person 2

        P1_img = Image.open(P1_path).convert('RGB')

        BP1_img = Image.open(BP1_path).convert('RGB')
        BP2_img = Image.open(BP2_path).convert('RGB')

        if self.use_inverse_seg:
            MP1_img = (np.load(MP1_path)-0.5)*2
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
            BP1 = self.transform(BP1_img)
            BP2 = self.transform(BP2_img)

            P1 = self.transform(P1_img)

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

        return {'P1': P1, 'BP1': BP1, 'P2': P1,'BP2': BP2,
                'P1_path': video_name + '0000.jpg',
                'P2_path': video_name + '{:04}.jpg'.format(image_i)}

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return 48600

    def name(self):
        return 'KeyDataset'
