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
        np.random.seed()
        random.seed()
        image = cv2.imread(join(self.data_dir, self.image_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = [image]

        if self.gt_list is not None:
            gt = cv2.imread(join(self.data_dir, self.gt_list[index]))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            data.append(gt)
    
        data = tuple(data)
        data = (self.transform(*data))

        if self.out_name:
            data = (*data, basename(self.image_list[index]))
        return data

    def __len__(self):
        return len(self.image_list)

    def _make_list(self, out_name):
        if self.phase=='train': 
            self.image_list = sorted(glob('/root/workspace/Challenge/LG/Datasets/256_cropped/train_256/*.png'))
            self.gt_list = sorted(glob('/root/workspace/Challenge/LG/Datasets/256_cropped/train_gt_256/*.png'))
        elif self.phase=='val' : 
            self.image_list = sorted(glob('/root/workspace/Challenge/LG/Datasets/valid_input/*.png'))
            self.gt_list = sorted(glob('/root/workspace/Challenge/LG/Datasets/valid_label/*.png'))
        else:
            self.image_list = sorted(glob('/root/workspace/Challenge/LG/Datasets/test_input/*.png'))
            self.gt_list = sorted(glob('/root/workspace/Challenge/LG/Datasets/test_input/*.png'))
        assert len(self.image_list)==len(self.gt_list), 'Input and GT length are not matched'
        