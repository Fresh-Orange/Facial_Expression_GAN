
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

class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling, padding_type=opt.padding_type)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(6, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_MM:
                self.netD_MM = networks.define_D(1, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)
                if opt.with_D_MM:
                    self.load_network(self.netD_MM, 'netD_MM', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            self.fake_MM_pool = ImagePool(opt.pool_size)
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
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_MM:
                self.optimizer_D_MM = torch.optim.Adam(self.netD_MM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            if opt.with_D_MM:
                self.optimizers.append(self.optimizer_D_MM)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
            if opt.with_D_MM:
                networks.print_network(self.netD_MM)
        print('-----------------------------------------------')
        self.pair_GANloss = 0
        self.loss_originL1 = 0
        self.loss_perceptual = 0
        self.pair_L1loss = 0

    def set_input(self, input):
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)

        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]


    def forward(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2 = self.netG(G_input)


    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)
        
        self.input_BP2 = Variable(self.input_BP2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2 = self.netG(G_input)


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_G(self):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = -torch.mean(pred_fake_PB)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = -torch.mean(pred_fake_PP)

        if self.opt.with_D_MM:
            pred_fake_MM = self.netD_MM(self.fake_p2[:,3:,:,:])
            self.loss_G_GAN_MM = self.criterionGAN(pred_fake_MM, True)

        # L1 loss
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            #print(losses[1])
            self.loss_originL1 = losses[1].item()#.data[0]
            #print(losses[2])
            self.loss_perceptual = losses[2].item()#.data[0]
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A


        pair_L1loss = self.loss_G_L1
        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB# * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP#*1.5# * self.opt.lambda_GAN * 1.5
                pair_GANloss = pair_GANloss * self.opt.lambda_GAN
                if self.opt.with_D_MM:
                    pair_GANloss += self.loss_G_GAN_MM * self.opt.lambda_GAN
                    pair_GANloss = pair_GANloss / 3
                else:
                    pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        pair_loss.backward()
        self.pair_L1loss = pair_L1loss.item()#.data[0]
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.item()#.data[0]
            self.grad_G_GANloss = pair_GANloss.grad


    def backward_D_basic(self, netD, real, fake, weight):
        # Real
        pred_real = netD(real)
        loss_D_real = -torch.mean(pred_real)# * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = torch.mean(pred_fake)# * self.opt.lambda_GAN

        # Compute loss for gradient penalty.
        alpha = torch.rand(real.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * real.data + (1 - alpha) * fake.data).requires_grad_(True)
        out_src = netD(x_hat)

        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # Combined loss
        loss_D = loss_D_real + loss_D_fake + d_loss_gp * weight
        # backward
        loss_D.backward()
        return loss_D, d_loss_gp*weight, loss_D.grad

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2.detach(), self.input_BP2), 1).data )
        loss_D_PB, loss_GP_PB, grad_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB, 10)
        self.loss_D_PB = loss_D_PB.item()#.data[0]
        self.loss_GP_PB = loss_GP_PB.item()

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2.detach(), self.input_P1), 1).data )
        loss_D_PP, loss_GP_PP, grad_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP, 10)
        self.loss_D_PP = loss_D_PP.item()#.data[0]
        self.loss_GP_PP = loss_GP_PP.item()

    # D: take(M, M') as input
    def backward_D_MM(self):
        real_MM = self.input_P1[:,3:,:,:]
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_MM = self.fake_MM_pool.query(self.fake_p2[:,3:,:,:].data)
        loss_D_MM = self.backward_D_basic(self.netD_MM, real_MM, fake_MM)
        self.loss_D_MM = loss_D_MM.item()  # .data[0]

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def optimize_parameters(self, ind=0):
        # forward
        self.forward()
        '''
        if ind % 5 == 0:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            print("hhh")
        '''
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()

        # D_MM
        if self.opt.with_D_MM:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_MM.zero_grad()
                self.backward_D_MM()
                self.optimizer_D_MM.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_MM:
            ret_errors['D_MM'] = self.loss_D_MM
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual


        ret_errors['GP_PP'] = self.loss_GP_PP
        ret_errors['GP_PB'] = self.loss_GP_PB
        '''
        ret_errors['GRAD_PP'] = self.grad_D_PP
        ret_errors['GRAD_PB'] = self.grad_D_PB
        ret_errors['GRAD_G_GANloss'] = self.grad_G_GANloss
        ret_errors['GRAD_G_Lloss'] = self.grad_G_Lloss
        '''
        return ret_errors

    def get_current_visuals(self):
        if self.opt.phase == 'test':
            vis = util.tensor2im(self.fake_p2.data)
            ret_visuals = OrderedDict([('vis', vis)])
        elif self.fake_p2.data.shape[1] == 4:
            height, width = self.input_P1.size(2), self.input_P1.size(3)
            input_P1 = util.tensor2im(self.input_P1.data[:, 0:3, :, :])
            input_P2 = util.tensor2im(self.input_P2.data[:, 0:3, :, :])

            input_BP1 = util.tensor2im(self.input_BP1.data)  # draw_pose_from_map(self.input_BP1.data)[0]
            input_BP2 = util.tensor2im(self.input_BP2.data)  # draw_pose_from_map(self.input_BP2.data)[0]

            fake_p2 = util.tensor2im(self.fake_p2.data[:, 0:3, :, :])

            mask = self.fake_p2.data[0][3].cpu()

            mask = np.where(mask > 0, 1, 0)
            mask = np.tile(mask, (3, 1, 1))
            mask = np.transpose(mask, (1, 2, 0))
            image = np.multiply(fake_p2, mask).astype(np.uint8)
            mask = (mask*255).astype(np.uint8)

            mask2 = self.input_P2.data[0][3].cpu()
            mask2 = np.where(mask2 > 0, 1, 0)
            mask2 = np.tile(mask2, (3, 1, 1))
            mask2 = np.transpose(mask2, (1, 2, 0))
            mask2 = (mask2 * 255).astype(np.uint8)


            vis = np.zeros((height, width * 8, 3)).astype(np.uint8)  # h, w, c
            vis[:, :width, :] = input_P1
            vis[:, width:width * 2, :] = input_BP1
            vis[:, width * 2:width * 3, :] = input_P2
            vis[:, width * 3:width * 4, :] = input_BP2
            vis[:, width * 4:width * 5, :] = mask2
            vis[:, width * 5:width * 6, :] = fake_p2
            vis[:, width * 6:width * 7, :] = mask
            vis[:, width * 7:, :] = image

            ret_visuals = OrderedDict([('vis', vis)])
        else:
            height, width = self.input_P1.size(2), self.input_P1.size(3)
            input_P1 = util.tensor2im(self.input_P1.data)
            input_P2 = util.tensor2im(self.input_P2.data)

            input_BP1 = util.tensor2im(self.input_BP1.data)  # draw_pose_from_map(self.input_BP1.data)[0]
            input_BP2 = util.tensor2im(self.input_BP2.data)  # draw_pose_from_map(self.input_BP2.data)[0]

            fake_p2 = util.tensor2im(self.fake_p2.data)

            vis = np.zeros((height, width * 5, 3)).astype(np.uint8)  # h, w, c
            vis[:, :width, :] = input_P1
            vis[:, width:width * 2, :] = input_BP1
            vis[:, width * 2:width * 3, :] = input_P2
            vis[:, width * 3:width * 4, :] = input_BP2
            vis[:, width * 4:, :] = fake_p2


            ret_visuals = OrderedDict([('vis', vis)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)
        if self.opt.with_D_MM:
            self.save_network(self.netD_MM, 'netD_MM', label, self.gpu_ids)

