import torch
from torch import nn
import torch.nn.functional as F

from models.gatedConv import GatedConv2d


class Downscale2d(nn.Module):
    def __init__(self, channels):
        super(Downscale2d, self).__init__()
        self.conv0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = GatedConv2d(channels, channels, kernelSize=3, stride=1, padding=1)
        self.res_block = ResidualBlock(channels, dilation=2)
    
    def forward(self, input):
        output = self.conv0(input)
        output = self.conv1(output)
        output = self.res_block(output)
        return output

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
class Upscale2d(nn.Module):
    def __init__(self, opt, inChannels, skChannels, outChannels):
        super(Upscale2d, self).__init__()
        self.opt = opt
        self.conv0 = nn.Sequential(
            nn.Conv2d(inChannels + skChannels, outChannels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(outChannels, outChannels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(outChannels // 8, outChannels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, input, skipConnection):
        output = torch.cat([F.interpolate(input, scale_factor=2, mode="nearest"), skipConnection], dim=1)
        output = self.conv0(F.dropout(output, self.opt.p, training=True))
        output = self.conv1(F.dropout(output, self.opt.p, training=True))
        attention = self.attention(output)
        output = output * attention
        return output
