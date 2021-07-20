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
        self.conv = ConvLayer(in_channels, growth_rate, stride=1, kernel_size=3)
        # self.innorm =  nn.InstanceNorm2d(growth_rate)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.act(self.conv(x))
        # out = self.act(self.innorm(self.conv(x)))
        out = torch.cat((x,out), dim=1)
        return out

# --- Build dense --- #
class VanilaModuleUp(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(VanilaModuleUp, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate, stride=1, kernel_size=3)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.act(self.conv(x))
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
            modules.append(VanilaModuleUp(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

# --- Build the Residual Dense Block --- #
class RDBUp(nn.Module):
    def __init__(self, in_channels):
        super(RDBUp, self).__init__()
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

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
        

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, y):
        x_up = F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear')
        x_up = self.conv1x1(x_up)
        out = x_up + y
        out = self.reflection_pad(out)
        out = self.conv2d(out)
        return out


# --- Main model  --- #
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv_e0 = ConvLayer(3, 16, kernel_size=11, stride=1) # H, W
        self.RDB__e0 = RDB(16)
        self.conv_e1 = ConvLayer(16, 32, kernel_size=3, stride=2) # H/2, W/2
        self.RDB__e1 = RDB(32)
        self.conv_e2 = ConvLayer(32, 64, kernel_size=3, stride=2) # H/4, W/4
        self.RDB__e2 = RDB(64)
        self.conv_e3 = ConvLayer(64, 128, kernel_size=3, stride=2) # H/8, W/8
        self.RDB__e3 = RDB(128)
        self.conv_e4 = ConvLayer(128, 256, kernel_size=3, stride=2) # H/16, W/16
        self.RDB__e4 = RDB(256)

        self.RDB__s1 = RDB(256)
        self.RDB__s2 = RDB(256)

        self.conv_d4 = UpsampleConvLayer(256, 128) # H/8, W/8
        self.RDB__d4 = RDBUp(128)
        self.out_d4 = nn.Conv2d(128, 3, kernel_size=1, padding=0)

        self.conv_d3 = UpsampleConvLayer(128, 64) # H/4, W/4
        self.RDB__d3 = RDBUp(64)
        self.out_d3 = nn.Conv2d(64, 3, kernel_size=1, padding=0)

        self.conv_d2 = UpsampleConvLayer(64, 32) # H/2, W/2
        self.RDB__d2 = RDBUp(32)
        self.out_d2 = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.conv_d1 = UpsampleConvLayer(32, 16) # H, W
        self.RDB__d1 = RDBUp(16)

        self.conv_out = ConvLayer(16, 3, kernel_size=3, stride=1)

    def forward(self, x):
        feat0 = self.RDB__e0(self.conv_e0(x))
        feat1 = self.RDB__e1(self.conv_e1(feat0))
        feat2 = self.RDB__e2(self.conv_e2(feat1))
        feat3 = self.RDB__e3(self.conv_e3(feat2))
        feat4 = self.RDB__e4(self.conv_e4(feat3))

        feat = self.RDB__s1(feat4)
        feat = self.RDB__s2(feat)

        feat = self.RDB__d4(self.conv_d4(feat, feat3))
        out4 = self.out_d4(feat)
        feat = self.RDB__d3(self.conv_d3(feat, feat2))
        out3 = self.out_d3(feat)
        feat = self.RDB__d2(self.conv_d2(feat, feat1))
        out2 = self.out_d2(feat)
        feat = self.RDB__d1(self.conv_d1(feat, feat0))
        out = self.conv_out(feat)
        return out, out2, out3, out4

class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride = 2, padding=1)

        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv1x1 = nn.Conv2d(512, 1, kernel_size=1, stride = 1, padding=0)

    def forward(self, x):
        feat1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        feat2 = F.leaky_relu(self.bn2(self.conv2(feat1)), negative_slope=0.2, inplace=True)
        feat3 = F.leaky_relu(self.bn3(self.conv3(feat2)), negative_slope=0.2, inplace=True)
        feat4 = F.leaky_relu(self.bn4(self.conv4(feat3)), negative_slope=0.2, inplace=True)
        prob = self.conv1x1(feat4)
        return prob
