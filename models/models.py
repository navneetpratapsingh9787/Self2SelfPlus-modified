import torch
from torch import nn
from torch.nn import init

from tqdm import tqdm

from models.autoencoder import DAE
from models.loss import *

from utils import utils

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
class Self2SelfPlus(nn.Module):
    def __init__(self, opt):
        super(Self2SelfPlus, self).__init__()
        self.opt = opt
        self.OKBLUE, self.ENDC = utils.bcolors.OKBLUE, utils.bcolors.ENDC
        self.net = DAE(opt)
        self.criterionIQA = IQALoss()
        self.criterionPerceptual = PerceptualLoss()
        self.computeNumParameter()
        utils.fixSeed(opt.seed)
        self.initializeNetwork()

    def computeLoss(self, noisyImage):
        loss = {}
        mask, denoisedImage = self.denoiseImage(noisyImage)

        # Self-Supervised Loss (L2)
        loss["Self-Supervised"] = torch.pow((denoisedImage - noisyImage) * (1 - mask), 2).sum() / (1 - mask).sum()
        # Perceptual Loss
        loss["Perceptual"] = self.criterionPerceptual(denoisedImage, noisyImage) * self.opt.lambdaPerceptual
        # IQA Loss
        loss["IQA"] = self.criterionIQA(denoisedImage) * self.opt.lambdaIQA

        return loss

    def forward(self, noisyImage, mode):
        if mode == "train":
            self.net.train()
            loss = self.computeLoss(noisyImage)
            return loss
        elif mode == "inference":
            self.net.eval()
            with torch.no_grad():
                finalImage = 0
                with tqdm(total=self.opt.numSample) as pBar:
                    for iter in range(1, self.opt.numSample + 1):
                        _, denoisedImage = self.denoiseImage(noisyImage)
                        finalImage += denoisedImage
                        pBar.set_description(desc=f"[{iter}/{self.opt.numSample}] < Saving Result! >")
                        pBar.update()
            return finalImage / self.opt.numSample
        else:
            raise NotImplementedError(f"{mode} is not supported")

    def denoiseImage(self, noisyImage):
        return self.net(noisyImage)

    def computeNumParameter(self):
        # Same as original
        networkList = [self.net]
        print(f"{self.OKBLUE}Self2Self+{self.ENDC}: Now Computing Model Parameters.")
        for network in networkList:
            numParameter = 0
            for _, module in network.named_modules():
                if isinstance(module, nn.Conv2d):
                    numParameter += sum([p.data.nelement() for p in module.parameters()])
            print(f"{self.OKBLUE}Self2Self+{self.ENDC}: {utils.bcolors.OKGREEN}[{network.__class__.__name__}]{self.ENDC} Total params : {numParameter:,}.")
        print(f"{self.OKBLUE}Self2Self+{self.ENDC}: Finished Computing Model Parameters.")

    def initializeNetwork(self):
        # Same as original
        def init_weights(m, initType=self.opt.initType, gain=0.02):
            className = m.__class__.__name__
            if hasattr(m, "weight") and className.find("Conv") != -1:
                if initType == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        networkList = [self.net]
        for network in networkList:
            network.apply(init_weights)

# Utility functions (assumed unchanged)
def assignOnGpu(opt, model):
    if opt.gpuIds != "-1":
        model = model.cuda()
    return model

def preprocessData(opt, noisyImage):
    if opt.gpuIds != "-1":
        noisyImage = noisyImage.cuda()
    return noisyImage
