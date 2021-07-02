import os
import numpy as np
import random
import torch
from PIL import Image

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
            img = Image.open(os.path.join(self.data_dir, self.img_list[index])).convert('RGB')
            gt  = Image.open(os.path.join(self.data_dir, self.gt__list[index])).convert('RGB')
            data = list(self.t_super(*[img, gt]))

        elif self.phase == 'val':
            img = Image.open(os.path.join(self.data_dir, self.img_list[index])).convert('RGB')
            gt  = Image.open(os.path.join(self.data_dir, self.gt__list[index])).convert('RGB')
            data = list(self.t_super(*[img, gt]))
            data.append(self.img_list[index])
        else : 
            img = Image.open(os.path.join(self.data_dir, self.img_list[index])).convert('RGB')
            data = list(self.t_super(*[img, img]))
            data.append(self.img_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.img_list)

    def _make_list(self, out_name):
        if self.phase == 'train':
            img_path = os.path.join('./lib/datasets/info', 'train_img.txt')
            gt__path = os.path.join('./lib/datasets/info', 'train_gt.txt')

            self.img_list = [line.strip() for line in open(img_path, 'r')][:50]
            self.gt__list = [line.strip() for line in open(gt__path, 'r')][:50]
        elif self.phase == 'val':            
            img_path = os.path.join('./lib/datasets/info', 'val_img.txt')
            gt__path = os.path.join('./lib/datasets/info', 'val_gt.txt')

            self.img_list = [line.strip() for line in open(img_path, 'r')][:2]
            self.gt__list = [line.strip() for line in open(gt__path, 'r')][:2]
        else :
            img_path = os.path.join('./lib/datasets/info', 'test_img.txt')
            self.img_list = [line.strip() for line in open(img_path, 'r')]
