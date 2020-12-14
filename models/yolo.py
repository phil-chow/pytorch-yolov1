import torch.nn as nn
from backbone import resnet18
from .modules import Conv2d, SPP, SAM





class YoloV1(nn.Module):
    def __init__(self, num_class=20, trainable=False):
        super(YoloV1, self).__init__()
        # 参数
        self.num_class = num_class
        # 是否是训练阶段
        self.trainable = trainable

        # YoloV1 network
        self.backbone = resnet18(pretrained=True)
        self.spp = SPP(512, 512)
        self.sam = SAM(512)

        self.conv_set = nn.Sequential(
            Conv2d(512, 256, kernel_size=1, leaky_relu=True),
            Conv2d(256, 512, kernel_size=3, padding=1, leaky_relu=True),
            Conv2d(512, 256, kernel_size=1, leaky_relu=True),
            Conv2d(256, 512, kernel_size=3, padding=1, leaky_relu=True)
        )
        self.pred = Conv2d(512, 1+self.num_class+4, kernel_size=1)

    def forward(self, x):
        # backbone
        c5 = self.backbone(x)
        # head
        c5 = self.spp(c5)
        c5 = self.sam(c5)
        c5 = self.conv_set(c5)
        # prediction
        prediction = self.pred(c5)
        prediction = prediction.view(c5.size(0), 1+self.num_class+4, -1).premute(0, 2, 1)
        # B:batch size, HW:H*W, C:5+num_class
        B, HW, C = prediction.size()
        # 置信度 [B, HW, 1]
        conf_pred = prediction[:,:,:1]
        # 预测类别 [B, HW, num_class]
        class_pred = prediction[:, :, 1:1+self.num_class]
        # 预测框 [B, HW, 4]
        txtytwth_pred = prediction[:, :, 1+self.num_class:]

        if self.trainable:
            pass
        else:
            pass

