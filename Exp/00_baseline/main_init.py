import argparse
import logging
import os
import threading
import time
import numpy as np
import shutil
from os.path import join, exists, split
from math import log10

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from dataset import RestList
import data_transforms as transforms
from main import RUN


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test']) #
    parser.add_argument('-d', '--data-dir', default=None, required=True) #
    parser.add_argument('-s', '--crop-size', default=0, type=int) #
    parser.add_argument('--step', type=int, default=200) #
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', #
                        help='input batch size for training (default: 64)') #
    parser.add_argument('--epochs', type=int, default=10, metavar='N', #
                        help='number of epochs to train (default: 10)') #
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)') #
    parser.add_argument('--resume', default='', type=str, metavar='PATH', #
                        help='path to latest checkpoint (default: none)') #
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    
    return args


def main():
    args = parse_args()
    
    dt_now = datetime.now()
    timeName = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(dt_now.year, dt_now.month, \
    dt_now.day, dt_now.hour, dt_now.minute)
    saveDirName = './runs/train/' + timeName
    if not os.path.exists(saveDirName):
        os.makedirs(saveDirName, exist_ok=True)

    # logging configuration
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(saveDirName + '/log_training.log')
    logger.addHandler(file_handler)

    RUN(args, saveDirName=saveDirName, logger=logger)

if __name__ == '__main__':
    main()
