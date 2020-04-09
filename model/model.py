import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.snconv2d import SNConv2d
from model.snlinear import SNLinear


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class NoTrackResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(NoTrackResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class BNResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(BNResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, track_running_stats=True, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, track_running_stats=True, momentum=0.5))

    def forward(self, x):
        return x + self.main(x)


class FInsNormResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(FInsNormResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out))

    def forward(self, x):
        return x + self.main(x)


class FastTrackResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(FastTrackResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True, momentum=0.8))

    def forward(self, x):
        return x + self.main(x)

class BigConvResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(BigConvResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=13, stride=1, padding=6, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=13, stride=1, padding=6, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class LandMarksDetect(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(LandMarksDetect, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x + Variable(torch.randn(x.size()).cuda() * 0.4)
        return self.main(x)


class FusionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(FusionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, full, mask, crop):
        x = torch.cat([full, mask, crop], dim=1)
        return self.main(x)

class FusionGenerater_V4(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=7):
        super(FusionGenerater_V4, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(3):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(3):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, true_full, last_full, mask, crop):
        x = torch.cat([true_full, last_full, mask, crop], dim=1)
        return self.main(x)

class RealFakeDiscriminator_V4(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=7):
        super(RealFakeDiscriminator_V4, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out_src = self.main(x)
        return out_src


class TwoPointBigConvExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(TwoPointBigConvExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(BigConvResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, oc, tc):
        x = torch.cat([x, oc, tc], dim=1)
        return self.main(x)

class BigConvExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(BigConvExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(BigConvResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class FastTrackExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(FastTrackExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True, momentum=0.8))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(FastTrackResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class FInsNormExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(FInsNormExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(FInsNormResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class BNExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(BNExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, track_running_stats=True, momentum=0.5))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim*2, track_running_stats=True, momentum=0.5))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(BNResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim//2, track_running_stats=True, momentum=0.5))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class NoTrackExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(NoTrackExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class ExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(ExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class AAExpressionGenerater(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(AAExpressionGenerater, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            #layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(AugmentedConv(in_channels=curr_dim, out_channels=curr_dim*2, kernel_size=4, dk=40, dv=4, Nh=1, relative=False, stride=2))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class RealFakeDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(RealFakeDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out_src = self.main(x)
        return out_src


class IdDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(IdDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(6, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        #layers.append(nn.Linear())
        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x, trans_x):
        x = torch.cat([x, trans_x], dim=1)
        #print(x.shape)
        out_src = self.main(x)
        #print(out_src.shape)
        return out_src


class KeypointDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(KeypointDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(6, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        #layers.append(nn.Linear())
        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x, kp):
        x = torch.cat([x, kp], dim=1)
        #print(x.shape)
        out_src = self.main(x)
        #print(out_src.shape)
        return out_src


class SNRealFakeDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(SNRealFakeDiscriminator, self).__init__()
        layers = []
        layers.append(SNConv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SNConv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(SNConv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out_src = self.main(x)
        return out_src


class SNIdDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(SNIdDiscriminator, self).__init__()
        layers = []
        layers.append(SNConv2d(6, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SNConv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(SNConv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))

        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x, trans_x):
        x = torch.cat([x, trans_x], dim=1)
        #print(x.shape)
        out_src = self.main(x)
        #print(out_src.shape)
        return out_src


class SNKeypointDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(SNKeypointDiscriminator, self).__init__()
        layers = []
        layers.append(SNConv2d(6, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SNConv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim * 2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        layers.append(SNConv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))

        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x, kp):
        x = torch.cat([x, kp], dim=1)
        #print(x.shape)
        out_src = self.main(x)
        #print(out_src.shape)
        return out_src

class FeatureExtractNet(nn.Module):
    def __init__(self):
        super(FeatureExtractNet, self).__init__()
        res50 = resnet50(pretrained=True)

        modules = list(res50.children())[:-1]
        self.model = nn.Sequential(*modules)


    def forward(self, x):
        return self.model(x)


class FaceModel(nn.Module):
    def __init__(self,embedding_size,num_classes,pretrained=False):
        super(FaceModel, self).__init__()

        self.model = resnet18(pretrained)

        self.embedding_size = embedding_size

        self.model.fc = nn.Linear(2048*4*4, self.embedding_size)

        self.model.classifier = nn.Linear(self.embedding_size, num_classes)


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        #print("x_size1", x.size())

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


from torch.nn.parameter import Parameter

class FaceModelCenter(nn.Module):
    def __init__(self,embedding_size,num_classes, checkpoint=None):
        super(FaceModelCenter, self).__init__()
        self.model = resnet18()
        self.model.avgpool = None
        self.model.fc1 = nn.Linear(512*3*3, 512)
        self.model.fc2 = nn.Linear(512, embedding_size)
        self.model.classifier = nn.Linear(embedding_size, num_classes)
        self.centers = torch.zeros(num_classes, embedding_size).type(torch.FloatTensor)
        self.num_classes = num_classes

        self.apply(self.weights_init)

        if checkpoint is not None:
            # Check if there are the same number of classes
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                self.load_state_dict(checkpoint['state_dict'])
                self.centers = checkpoint['centers']
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if "classifier" not in name:
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


    def get_center_loss(self,target, alpha):
        batch_size = target.size(0)
        features_dim = self.features.size(1)

        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)

        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0,target_expand).cuda()

        criterion = nn.MSELoss()
        center_loss = criterion(self.features,  centers_batch)

        diff = centers_batch - self.features

        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)

        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))

        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)

        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)

        diff_cpu = alpha * diff_cpu

        for i in range(batch_size):
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss, self.centers

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        #feature for center loss
        x = self.model.fc2(x)
        self.features = x
        self.features_norm = self.l2_norm(x)
        return self.features_norm

    def forward_classifier(self,x):
        features_norm = self.forward(x)
        x = self.model.classifier(features_norm)
        return F.log_softmax(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, use_BN = False, downsample=False):
        super(ResBlock, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(in_channels, out_channels, hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)




class SNResRealFakeDiscriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=4):
        super(SNResRealFakeDiscriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf*16, 1))
    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(3, ndf)]
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)
    def forward(self, x):
        out = self.res_d(x)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, 1024)
        return self.fc(out)


class SNResIdDiscriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=4):
        super(SNResIdDiscriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf*16, 1))
    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(6, ndf)]  # input_channel = 6
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)
    def forward(self, x, trans_x):
        x = torch.cat([x, trans_x], dim=1)
        out = self.res_d(x)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, 1024)
        return self.fc(out)


class SNResKeypointDiscriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=4):
        super(SNResKeypointDiscriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf*16, 1))
    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(6, ndf)]
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)
    def forward(self, x, kp):
        x = torch.cat([x, kp], dim=1)
        out = self.res_d(x)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, 1024)
        return self.fc(out)

class ResBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
        super(ResBlockG, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
    def forward_residual_connect(self, input):
        out = self.conv_sc(input)
        if self.upsample:
             out = self.upsampling(out)
            #out = self.upconv2(out)
        return out
    def forward(self, input):
        out = self.relu(self.bn1(input))
        out = self.conv1(out)
        if self.upsample:
             out = self.upsampling(out)
             #out = self.upconv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out_res = self.forward_residual_connect(input)
        return out + out_res

class SNResGenerator(nn.Module):
    def __init__(self, ngf, z=128, nlayers=4):
        super(SNResGenerator, self).__init__()
        self.input_layer = nn.Linear(z, (4 ** 2) * ngf * 16)
        self.generator = self.make_model(ngf, nlayers)

    def make_model(self, ngf, nlayers):
        model = []
        tngf = ngf*16
        for i in range(nlayers):
            model += [ResBlockG(tngf, tngf/2, upsample=True)]
            tngf /= 2
        model += [nn.BatchNorm2d(ngf)]
        model += [nn.ReLU()]
        model += [nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)

    def forward(self, x, kp):
        x = torch.cat([x, kp], dim=1)
        out = self.input_layer(x)
        out = out.view(x.size(0), -1, 4, 4)
        out = self.generator(out)

        return out