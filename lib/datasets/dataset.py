import os
import numpy as np
import random
import torch
from os.path import join
from PIL import Image
from glob import glob

import lib.datasets.transforms as transforms


class RestList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, t_super, out_name=False):
        self.data_dir = data_dir
        self.phase = phase
        self.t_super = t_super
        self.out_name = out_name

        self.img_list = None
        self.gt__list = None

        self._make_list(out_name)

    def __getitem__(self, index):
        np.random.seed()
        random.seed()

        if self.phase == 'train' :
            img = Image.open(self.img_list[index]).convert('RGB')
            gt  = Image.open(self.gt__list[index]).convert('RGB')
            data = list(self.t_super(*[img, gt]))

        elif self.phase == 'val':
            img = Image.open(self.img_list[index]).convert('RGB')
            gt  = Image.open(self.gt__list[index]).convert('RGB')
            data = list(self.t_super(*[img, gt]))
            data.append(self.img_list[index])
        else : 
            img = Image.open(self.img_list[index]).convert('RGB')
            data = list(self.t_super(*[img, img]))
            data.append(self.img_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.img_list)

    def _make_list(self, out_name):
        if self.phase == 'train':

            self.img_list = glob(self.data_dir + '/train_data/*.png')
            self.gt__list = glob(self.data_dir + '/train_gt/*.png')

        elif self.phase == 'val':            
            
            self.img_list = glob(self.data_dir + '/val_data/*.png')
            self.gt__list = glob(self.data_dir + '/val_gt/*.png')

        else :
            self.img_list = glob(self.data_dir + '/test_data/*.png')
