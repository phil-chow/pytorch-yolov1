import torch
import torch.nn as nn
import numpy as np
from backbone import resnet18
from .modules import Conv2d, SPP, SAM


class YoloV1(nn.Module):
    def __init__(self, input_size=None, num_class=20, trainable=False):
        super(YoloV1, self).__init__()
        # 参数
        self.num_class = num_class
        # 是否是训练阶段
        self.trainable = trainable
        self.stride = 32
        self.input_size = input_size
        self.grid_cell = self.create_grid()
        self.scale = torch.tensor([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])

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
        self.pred = Conv2d(512, 1 + self.num_class + 4, kernel_size=1)

    def create_grid(self):
        w, h = self.input_size[1], self.input_size[0]
        ws, hs = w // self.stride, h // self.stride
        grid_x, grid_y = torch.meshgrid([torch.arange(ws), torch.arange(hs)])
        grid_xy = torch.stack([grid_x, grid_y], dim=2)
        grid_xy = grid_xy.view(1, ws * hs, 2)
        return grid_xy

    def decode_bbox(self, txtytwth_pred):
        """
        获取预测的bbox参数
        :param txtytwth_pred:[tx,ty,tw,th]
        :return: output[xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(txtytwth_pred)
        txtytwth_pred[:, :, :2] = torch.sigmoid(txtytwth_pred[:, :, :2]) + self.grid_cell
        txtytwth_pred[:, :, 2:] = torch.exp(txtytwth_pred[:, :, 2:])

        output[:, :, 0] = txtytwth_pred[:, :, 0] * self.stride - txtytwth_pred[:, :, 2] / 2
        output[:, :, 1] = txtytwth_pred[:, :, 1] * self.stride - txtytwth_pred[:, :, 3] / 2
        output[:, :, 2] = txtytwth_pred[:, :, 0] * self.stride + txtytwth_pred[:, :, 2] / 2
        output[:, :, 3] = txtytwth_pred[:, :, 1] * self.stride + txtytwth_pred[:, :, 3] / 2

        return output

    def nms(self, det, thresh):
        x1 = det[:, 0]
        y1 = det[:, 1]
        x2 = det[:, 2]
        y2 = det[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        scores = det[:, 4]
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xxmin = np.maximum(x1[i], x1[order[1:]])
            yymin = np.maximum(y1[i], y1[order[1:]])
            xxmax = np.minimum(x2[i], x2[order[1:]])
            yymax = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xxmax - xxmin)
            h = np.maximum(0, yymax - yymin)

            overlap = w * h
            ious = overlap / (areas[i] + areas[order[1:]] - overlap)
            idx = np.where(ious <= thresh)[0]
            order = order[idx + 1]
        return keep

    def forward(self, x):
        # backbone
        c5 = self.backbone(x)
        # head
        c5 = self.spp(c5)
        c5 = self.sam(c5)
        c5 = self.conv_set(c5)
        # prediction
        prediction = self.pred(c5)
        prediction = prediction.view(c5.size(0), 1 + self.num_class + 4, -1).premute(0, 2, 1)
        # B:batch size, HW:H*W, C:num_class+5
        B, HW, C = prediction.size()
        # 置信度 [B, HW, 1]
        conf_pred = prediction[:, :, :1]
        # 预测类别 [B, HW, num_class]
        class_pred = prediction[:, :, 1:1 + self.num_class]
        # 预测框 [B, HW, 4]
        txtytwth_pred = prediction[:, :, 1 + self.num_class:]

        if self.trainable:
            pass
        else:
            pass
