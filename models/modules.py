import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, leaky_relu=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPP, self).__init__()
        self.fuse_conv = Conv2d(in_channels * 4, out_channels, 1, leaky_relu=True)

    def forward(self, x):
        x1 = F.max_pool2d(x, 13, stride=1, padding=6)
        x2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x3 = F.max_pool2d(x, 5, stride=1, padding=2)
        x = torch.cat([x, x1, x2, x3], dim=1)
        return self.fuse_conv(x)


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv(x)
        return x1 * x
