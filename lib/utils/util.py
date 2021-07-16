import os
import threading
import numpy as np
import shutil
import math
import json
# from math import log10, exp
from PIL import Image
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models import vgg16
from torchvision.utils import save_image


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight_data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def psnr(true, pred, pixel_max):
    """
    Computes the PSNR
    """
    score = 20*np.log10(pixel_max/rmse_score(true, pred))
    return score

def rmse_score(true, pred):
    score = torch.sqrt(torch.mean((true-pred)**2))
    return score

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(args, epoch, optimizer, lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    # """
    if args.lr_mode == 'step':
        lr = max(args.lr * (0.5 ** (epoch // args.step)), 1e-10)
    # elif args.lr_mode == 'poly':
    #     lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    # else:
    #     raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_output_images(predictions, pre, pathes, output_dir, epoch, phase):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(pathes)):
        os.makedirs(output_dir, exist_ok=True)
        if phase == 'val':
            fn = os.path.join(output_dir, pathes[ind].split('/')[-1].replace('.png', '.jpg'))
        elif phase == 'test':
            fn = os.path.join(output_dir, pathes[ind].split('/')[-1])
        else:
            raise ValueError('No such phase,')

        save_image(predictions[ind], fn)
        
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, sigma = 1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def Gaussiansmoothing(img, channel=3, window_size = 11):
    window = create_window(window_size, channel, sigma=5)
    
    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    x_smooth = F.conv2d(img, window, padding = window_size//2, groups = channel)
    
    return x_smooth, img - x_smooth

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_losses(iters, losses, path):
    plt.plot(iters, losses[0], 'r', label='Total loss')
    plt.plot(iters, losses[1], 'b', label='Base loss')
    plt.plot(iters, losses[2], 'g', label='GAN loss')
    plt.legend(loc='upper right')   
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.grid()
    plt.savefig(path)
    plt.clf()
    plt.cla()

def plot_scores(epochs, scores, path):
    plt.plot(epochs, scores, 'r')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(10,35,step=5))
    plt.ylabel('Scores')
    plt.grid()
    plt.savefig(path)
    plt.clf()
    plt.cla()
    with open(path.replace('.jpg', '.json'), 'w') as fp:
        json.dump([epochs, scores], fp)

def plot_lrs(epochs, lrs, path):
    plt.plot(epochs, lrs[0], 'r', label='Generator')
    plt.plot(epochs, lrs[1], 'b', label='Discriminator')
    plt.legend(loc='upper right')   
    plt.xlabel('Epochs')
    plt.ylabel('lrs')
    plt.grid()
    plt.savefig(path)
    plt.clf()
    plt.cla()