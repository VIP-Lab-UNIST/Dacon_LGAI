from os.path import join, basename, exists
import numpy as np
import random
import torch
from os.path import join
from PIL import Image
from glob import glob
import cv2
import lib.datasets.transforms as transforms


class RestList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform, out_name=False):
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform
        self.out_name = out_name

        self.image_list = None
        self.gt_list = None

        self._make_list(out_name)

    def __getitem__(self, index):
        # np.random.seed()
        # random.seed()

        image = cv2.imread(join(self.data_dir, self.image_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = []
        # only training data
        division = 512
        value_w = image.shape[0]//division
        remainder_w = image.shape[0]%division

        value_h = image.shape[1]//division
        remainder_h = image.shape[1]%division

        image_up_left = image[0:value_w*division, 0:value_h*division, :]
        image_up_right = image[0:value_w*division, remainder_h:image.shape[1], :]
        
        image_down_left = image[remainder_w:image.shape[0], 0:value_h*division , :]
        image_down_right = image[remainder_w:image.shape[0], remainder_h:image.shape[1], :]

        gt = cv2.imread(join(self.data_dir, self.gt_list[index]))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        data.extend([image_up_left, image_up_right, image_down_left, image_down_right, gt])

        data = tuple(data)
        # print('**************************')
        # print('index: ', index)
        # print('name: ', basename(self.image_list[index]))
        # print('image.shape: ', image.shape)
        # print('value_w: ', value_w, 'remainder_w', remainder_w)
        # print('value_h: ', value_h, 'remainder_h', remainder_h)
        # print('image_up_left.shape: ', image_up_left.shape)
        # print('image_up_right.shape: ', image_up_right.shape)
        # print('image_down_left.shape: ', image_down_left.shape)
        # print('image_down_right.shape: ', image_down_right.shape)
        data = (self.transform(*data))

        if self.out_name:
            data = (*data, basename(self.image_list[index]))

        return data

    def __len__(self):
        return len(self.image_list)

    def _make_list(self, out_name):
        if self.phase=='train': 
            self.image_list = sorted(glob(join(self.data_dir, 'train_input/*.png')))
            self.gt_list = sorted(glob(join(self.data_dir, 'train_input/*.png')))
        elif self.phase=='val' : 
            self.image_list = sorted(glob(join(self.data_dir, 'valid_input/*.png')))
            self.gt_list = sorted(glob(join(self.data_dir, 'valid_label/*.png')))
        else:
            self.image_list = sorted(glob(join(self.data_dir, 'test_input/*.png')))
            self.gt_list = sorted(glob(join(self.data_dir, 'test_input/*.png')))
        assert len(self.image_list)==len(self.gt_list), 'Input and GT length are not matched'
        
        