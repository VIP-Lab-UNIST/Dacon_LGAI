import time
import datetime
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

# --- Build dense --- #
class VanilaModule(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(VanilaModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(growth_rate, affine=True)

    def forward(self, x):
        out = F.relu(self.norm(self.conv(x)))
        out = torch.cat((x,out), dim=1)
        return out

# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels):
        super(RDB, self).__init__()
        _in_channels = in_channels
        growth_rate = in_channels // 2
        modules = []
        for i in range(4):
            modules.append(VanilaModule(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

# --- Main model  --- #
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=1)
        self.RDB_enc = RDB(64)
        self.RDB_dec = RDB(64)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        feat = self.conv_in(x)
        feat = self.RDB_enc(feat)
        feat = self.RDB_dec(feat)        
        out = x + self.conv_out(feat)
        return out
