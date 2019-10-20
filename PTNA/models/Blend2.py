import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

class BlendModel(BaseModel):
    def name(self):
        return 'Blend'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_P_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_PB_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BG_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_GT_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.mask_person_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.mask_bg_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_seg_P_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_seg_BG_set = self.Tensor(nb, opt.P_input_nc, size, size)

        self.netG = networks.define_G(opt.P_input_nc*3, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD1 = networks.define_D(3, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)
            self.netD2 = networks.define_D(3, opt.ndf,
                                           opt.which_model_netD,
                                           opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                           not opt.no_dropout_D,
                                           n_downsampling=opt.D_n_downsampling)
            self.netD_PB = networks.define_D(6, opt.ndf,
                                           opt.which_model_netD,
                                           opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                           not opt.no_dropout_D,
                                           n_downsampling=opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                self.load_network(self.netD1, 'netD1', which_epoch)
                self.load_network(self.netD2, 'netD2', which_epoch)
                self.load_network(self.netD_PB, 'netD_PB', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_D1_pool = ImagePool(opt.pool_size)
            self.fake_D2_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Excption('Unsurportted type of L1!')
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)
            self.optimizers.append(self.optimizer_D_PB)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD1)
            networks.print_network(self.netD2)
            networks.print_network(self.netD_PB)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_P = input['P']
        input_PB = input['PB']
        input_BG = input['BG']
        input_GT = input['GT']
        mask_person = input["MP"]
        mask_bg = input["MBG"]
        input_seg_p = input["SP"]
        input_seg_bg = input["SBG"]

        self.input_P_set.resize_(input_P.size()).copy_(input_P)
        self.input_PB_set.resize_(input_PB.size()).copy_(input_PB)
        self.input_BG_set.resize_(input_BG.size()).copy_(input_BG)

        self.input_GT_set.resize_(input_GT.size()).copy_(input_GT)
        self.mask_person_set.resize_(mask_person.size()).copy_(mask_person)
        self.mask_bg_set.resize_(mask_bg.size()).copy_(mask_bg)
        self.input_seg_P_set.resize_(input_seg_p.size()).copy_(input_seg_p)
        self.input_seg_BG_set.resize_(input_seg_bg.size()).copy_(input_seg_bg)

        ###########################################
        self.image_paths = input['P_path'][0] + '___' + input['BG_path'][0]


    def forward(self):
        self.input_P = Variable(self.input_P_set)
        self.input_PB = Variable(self.input_PB_set)
        self.input_BG = Variable(self.input_BG_set)
        self.input_GT = Variable(self.input_GT_set)
        self.mask_person = Variable(self.mask_person_set)
        self.mask_bg = Variable(self.mask_bg_set)
        self.input_seg_P = Variable(self.input_seg_P_set)
        self.input_seg_BG = Variable(self.input_seg_BG_set)

        G_input = torch.cat((self.input_PB, self.input_P, self.input_BG), 1)
        self.fake = self.netG(G_input)


    def test(self):
        self.input_P = Variable(self.input_P_set)
        self.input_PB = Variable(self.input_PB_set)
        self.input_BG = Variable(self.input_BG_set)

        G_input = torch.cat((self.input_PB_set, self.input_P_set, self.input_BG_set), 1)
        self.fake = self.netG(G_input)


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_G(self):
        pred_fake_D1 = self.netD1(self.fake)
        self.loss_G_D1 = self.criterionGAN(pred_fake_D1, True)

        pred_fake_D2 = self.netD2(self.fake)
        self.loss_G_D2 = self.criterionGAN(pred_fake_D2, True)

        pred_fake_PB = self.netD_PB(torch.cat((self.input_PB, self.fake), 1))
        self.loss_G_D_PB =self.criterionGAN(pred_fake_PB, True)

        seg_P = self.fake.mul(self.mask_person)
        seg_BG = self.fake.mul(self.mask_bg)

        input_seg_P_t = self.input_seg_P.mul(self.mask_person)
        input_seg_BG_t = self.input_seg_BG.mul(self.mask_bg)


        self.loss_GP = self.criterionL1(seg_P, input_seg_P_t) * self.opt.lambda_A * 1.5
        self.loss_GBG = self.criterionL1(seg_BG, input_seg_BG_t) * self.opt.lambda_A

        loss = self.loss_GP + self.loss_GBG + (self.loss_G_D1 + self.loss_G_D2) / 2 * self.opt.lambda_GAN + self.loss_G_D_PB * self.opt.lambda_GAN

        loss.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D1(self):
        real = self.input_GT
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake = self.fake_D1_pool.query(self.fake.data)
        loss_D1 = self.backward_D_basic(self.netD1, real, fake)
        self.loss_D1 = loss_D1.item()#.data[0]

    def backward_D2(self):
        real = self.input_GT
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake = self.fake_D2_pool.query(self.fake.data)
        loss_D2 = self.backward_D_basic(self.netD2, real, fake)
        self.loss_D2 = loss_D2.item()#.data[0]


    def backward_D_PB(self):
        real_PB = torch.cat((self.input_PB, self.input_GT), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PB = self.fake_PB_pool.query(torch.cat((self.input_PB, self.fake), 1).data)
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.item()#.data[0]


    def optimize_parameters(self):
        # forward
        self.forward()
        # D1
        for i in range(self.opt.DG_ratio):
            self.optimizer_D1.zero_grad()
            self.backward_D1()
            self.optimizer_D1.step()
        # D2
        for i in range(self.opt.DG_ratio):
            self.optimizer_D2.zero_grad()
            self.backward_D2()
            self.optimizer_D2.step()

        # DPB
        for i in range(self.opt.DG_ratio):
            self.optimizer_D_PB.zero_grad()
            self.backward_D_PB()
            self.optimizer_D_PB.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['D1'] = self.loss_D1
        ret_errors['D2'] = self.loss_D2
        ret_errors['D_PB'] = self.loss_D_PB
        ret_errors['G_P'] = self.loss_GP.item()
        ret_errors['G_BG'] = self.loss_GBG.item()
        return ret_errors

    def get_current_visuals(self):
        if self.opt.phase == 'test':
            vis = util.tensor2im(self.fake.data)
            ret_visuals = OrderedDict([('vis', vis)])
        else:
            height, width = self.input_P.size(2), self.input_P.size(3)
            input_BG = util.tensor2im(self.input_BG.data)
            input_PB = util.tensor2im(self.input_PB.data)

            input_P = util.tensor2im(self.input_P.data)  # draw_pose_from_map(self.input_BP1.data)[0]
            input_GT = util.tensor2im(self.input_GT.data)  # draw_pose_from_map(self.input_BP2.data)[0]

            fake = util.tensor2im(self.fake.data)

            input_seg_P = util.tensor2im(self.input_seg_P.data)  # draw_pose_from_map(self.input_BP1.data)[0]
            input_seg_BG = util.tensor2im(self.input_seg_BG.data)  # draw_pose_from_map(self.input_BP2.data)[0]




            vis = np.zeros((height, width * 7, 3)).astype(np.uint8)  # h, w, c
            vis[:, :width, :] = input_BG
            vis[:, width:width * 2, :] = input_PB
            vis[:, width * 2:width * 3, :] = input_P
            vis[:, width * 3:width * 4, :] = input_GT
            vis[:, width * 4:width * 5, :] = input_seg_P
            vis[:, width * 5:width * 6, :] = input_seg_BG
            vis[:, width * 6:, :] = fake
            ret_visuals = OrderedDict([('vis', vis)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)

        self.save_network(self.netD1,  'netD1',  label, self.gpu_ids)

        self.save_network(self.netD2, 'netD2', label, self.gpu_ids)

        self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids)

