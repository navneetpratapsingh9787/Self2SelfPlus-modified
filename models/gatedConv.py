from torch import nn


class GatedConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding):
        super(GatedConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, padding_mode="reflect")),
            nn.LeakyReLU(0.2)
        )
        self.maskedConv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, padding_mode="reflect")),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.conv(input)
        outputMask = self.maskedConv(input)
        return output * outputMask



