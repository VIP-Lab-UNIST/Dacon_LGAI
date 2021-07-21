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
    def __init__(self, ssim_weight, perc_weight, regular_weight):
        super(LossFunction, self).__init__()
        vgg_model = vgg16(pretrained=True).features
        vgg_model = vgg_model.cuda()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_module = VGG(vgg_model)
        self.ssim_module = SSIM()
        self.CR = CR()
        self.ssim_weight = ssim_weight
        self.perc_weight = perc_weight
        self.regular_weight = regular_weight

    def forward(self, input_img, out_img, gt_img):
        l1_loss = F.l1_loss(out_img, gt_img)
        ssim_loss = self.ssim_module(out_img, gt_img)
        p_loss = []
        inp_features, pv = self.vgg_module(out_img)
        gt_features, _ = self.vgg_module(gt_img)
        for i in range(3):
            p_loss.append(F.l1_loss(inp_features[i],gt_features[i]))
        perc_loss = sum(p_loss)/len(p_loss)
        cr_regularizer = self.CR(input_img, out_img, gt_img)

        return l1_loss + self.ssim_weight*ssim_loss + self.perc_weight*perc_loss + self.regular_weight*cr_regularizer

# --- Contrastive Regularization  --- #
class CR(torch.nn.Module):
    def __init__(self):
        super(CR, self).__init__()
        self.vgg_layers = vgg16(pretrained=True).features.cuda()
        self.layer_output = [1,3,5,9,13]
        self.layer_weight = [1/32, 1/16, 1/8, 1/4, 1]
        
    def extract_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if int(name) in self.layer_output:
                output.append(x)
        return output

    def forward(self, inputs, outs, gts):
        
        regularization = torch.zeros(5)
        negative = self.extract_features(inputs)
        anchor = self.extract_features(outs)
        positive = self.extract_features(gts)
        for i, (weight, neg, anch, pos) in enumerate(zip(self.layer_weight, negative, anchor, positive)):
            regularization[i] = ( weight*(F.l1_loss(pos, anch)/F.l1_loss(neg, anch)) )

        return torch.sum(regularization)

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

