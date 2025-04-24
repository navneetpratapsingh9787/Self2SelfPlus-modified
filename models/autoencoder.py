import torch
from torch import nn
import torch.nn.functional as F

from models.gatedConv import GatedConv2d
from models.architecture import Downscale2d, Upscale2d
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.norm = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        return out + residual

class DAE(nn.Module):
    def __init__(self, opt):
        super(DAE, self).__init__()
        self.opt = opt
        channels = opt.channels

        # Encoder with Residual Blocks
        self.EB0 = nn.Sequential(
            GatedConv2d(opt.inputDim * 2, channels, kernelSize=3, stride=1, padding=1),
            ResidualBlock(channels, dilation=1),
            GatedConv2d(channels, channels, kernelSize=3, stride=1, padding=1)
        )
        self.EB1 = Downscale2d(channels)
        self.EB2 = Downscale2d(channels)
        self.EB3 = Downscale2d(channels)
        self.EB4 = Downscale2d(channels)
        self.EB5 = Downscale2d(channels)

        # Decoder
        self.DB0 = Upscale2d(opt, channels, channels, channels * 2)
        self.DB1 = Upscale2d(opt, channels * 2, channels, channels * 2)
        self.DB2 = Upscale2d(opt, channels * 2, channels, channels * 2)
        self.DB3 = Upscale2d(opt, channels * 2, channels, channels * 2)
        self.DB4 = Upscale2d(opt, channels * 2, channels, channels * 2)
        self.DB5 = nn.Conv2d(channels * 2, opt.inputDim, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, input):
        n, c, h, w = input.size()
        p = self.opt.p if self.training else 0.5  # Adaptive dropout
        mask = F.dropout(torch.ones((n, c, h, w), device=input.device), p, True) * (1 - p)
        input = (mask * input).detach()

        # Encoder
        e0 = self.EB0(torch.cat([mask, input], dim=1))
        e1 = self.EB1(e0)
        e2 = self.EB2(e1)
        e3 = self.EB3(e2)
        e4 = self.EB4(e3)
        e5 = self.EB5(e4)

        # Decoder
        d1 = self.DB0(e5, e4)
        d2 = self.DB1(d1, e3)
        d3 = self.DB2(d2, e2)
        d4 = self.DB3(d3, e1)
        d5 = self.DB4(d4, e0)
        d6 = self.DB5(d5)
        return mask, d6
