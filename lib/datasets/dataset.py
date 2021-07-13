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

        if self.phase == 'train':
            gt = cv2.imread(join(self.data_dir, self.gt_list[index]))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(join(self.data_dir, self.mask_list[index]), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
            data.extend([gt, mask])
        elif self.phase == 'val':
            gt = cv2.imread(join(self.data_dir, self.gt_list[index]))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            data.append(gt)
        else:
            pass

        data = tuple(data)
        data = (self.transform(*data))

        if self.out_name:
            data = (*data, basename(self.image_list[index]))

        return data

    def __len__(self):
        return len(self.image_list)

    def _make_list(self, out_name):
        if self.phase=='train': 
            self.image_list = sorted(glob(join(self.data_dir, 'train_256/*.png')))
            self.gt_list = sorted(glob(join(self.data_dir, 'train_gt_256/*.png')))
            self.mask_list = sorted(glob(join(self.data_dir, 'train_256_mask/*.png')))
            assert len(self.image_list)==len(self.gt_list), 'Input and GT length are not matched'
        elif self.phase=='val' : 
            self.image_list = sorted(glob(join(self.data_dir, 'valid_input_img/*.png')))
            self.gt_list = sorted(glob(join(self.data_dir, 'valid_label_img/*.png')))
            assert len(self.image_list)==len(self.gt_list), 'Input and GT length are not matched'
        else:
            self.image_list = sorted(glob(join(self.data_dir, 'test_input_img/*.png')))
            self.gt_list = sorted(glob(join(self.data_dir, 'test_input_img/*.png')))
            assert len(self.image_list)==len(self.gt_list), 'Input and GT length are not matched'
        