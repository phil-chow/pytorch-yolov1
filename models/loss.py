import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, inputs, target):
        super(MSELoss, self).__init__()
