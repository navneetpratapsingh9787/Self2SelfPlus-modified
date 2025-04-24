import torch
import torch.nn as nn
import pyiqa
from torchvision.models import vgg16

class IQALoss(nn.Module):
    def __init__(self):
        super(IQALoss, self).__init__()
        self.model = pyiqa.create_metric("paq2piq", as_loss=True)
    
    def forward(self, input):
        return torch.pow(100 - self.model(input), 2).mean()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights='IMAGENET1K_V1').features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        return self.criterion(x_vgg, y_vgg)
