import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()


def loss(pred_conf, pred_class, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    conf_loss_function = MSELoss()
    class_loss_function = nn.CrossEntropyLoss(reduction="none")
    txty_loss_function = nn.BCEWithLogitsLoss(reduction="none")
    twth_loss_function = nn.MSELoss(reduction="none")

    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    # 转成[B, num_class, H*W]
    pred_class = pred_class.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
