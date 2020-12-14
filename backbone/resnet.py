import torch
import torch.nn as nn


def conv_3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv_1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        indentify = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            indentify = self.downsample(indentify)
        out += indentify
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layers(block, 64, layers[0])
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)

    def forward(self, x):
        c_1 = self.conv1(x)
        c_1 = self.bn1(c_1)
        c_1 = self.maxpool(c_1)

        c_2 = self.layer1(c_1)
        c_3 = self.layer2(c_2)
        c_4 = self.layer3(c_3)
        c_5 = self.layer4(c_4)
        return c_5

    def _make_layers(self, block, channels, layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(conv_1x1(self.in_channels, channels, stride=stride),
                                       nn.BatchNorm2d(channels))
        layers = []
        layers.append(block(self.in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, layer):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


if __name__ == '__main__':
    resnet18_model = resnet18()
    print(resnet18_model)
