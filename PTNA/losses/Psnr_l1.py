from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

class Psnr_l1Loss(nn.Module):
    def __init__(self, lambda_L1, lambda_psnr, perceptual_layers, gpu_ids):
        super(Psnr_l1Loss, self).__init__()

        self.lambda_L1 = lambda_L1
        self.gpu_ids = gpu_ids
        self.psnr = lambda_psnr

    def forward(self, inputs, targets):
        if self.lambda_L1 == 0 and self.lambda_perceptual == 0:
            return Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)), Variable(torch.zeros(1))
        # normal L1
        loss_l1 = F.l1_loss(inputs, targets) * self.lambda_L1

        mse = torch.mean((inputs - targets) ** 2) + 0.000000001
        loss_psnr = 1-torch.log10(1 / torch.sqrt(mse)) * self.psnr/ 10

        loss = loss_l1 + loss_psnr

        return loss, loss_l1, loss_psnr

