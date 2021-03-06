import os
import threading
import numpy as np
import shutil
import math
import logging
from PIL import Image
from datetime import datetime
# from math import log10, exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchvision.models import vgg16

from lib.utils.util import create_window, _ssim


class LossFunction(torch.nn.Module):
    def __init__(self, weight_ssim, weight_perc):
        super(LossFunction, self).__init__()
        vgg_model = vgg16(pretrained=True).features
        vgg_model = vgg_model.cuda()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.ssim_module = SSIM()
        self.weight_ssim = weight_ssim
        self.vgg_module = VGG(vgg_model)
        self.weight_perc = weight_perc


    def forward(self, out_img, gt_img):
        sm_l1_loss = F.smooth_l1_loss(out_img, gt_img)
        # mse_loss = F.mse_loss(out_img, gt_img)
        ssim_loss = self.ssim_module(out_img, gt_img)
        p_loss = []
        inp_features, pv = self.vgg_module(out_img)
        gt_features, _ = self.vgg_module(gt_img)
        for i in range(3):
            p_loss.append(F.mse_loss(inp_features[i],gt_features[i]))
        perc_loss = sum(p_loss)/len(p_loss)

        return sm_l1_loss + self.weight_ssim*ssim_loss + self.weight_perc*perc_loss
        # return mse_loss + ssim_loss + 0.01 * perc_loss

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return (1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average))

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# --- Perceptual loss network  --- #
class VGG(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGG, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        
    def extract_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output, x

    def forward(self, x):
        return self.extract_features(x)