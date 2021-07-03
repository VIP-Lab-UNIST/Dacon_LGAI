from os.path import join, basename, exists
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

        image = cv2.imread(join(self.data_dir, self.phase, self.image_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = [image]

        if self.gt_list is not None:
            gt = cv2.imread(join(self.data_dir, self.phase, self.gt_list[index]))
            gt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data.append(gt)
    
        data = tuple(data)
        data = (self.transforms(*data))

        if self.out_name:
            data = (*data, basename(self.image_list[index]))

        return data

    def __len__(self):
        return len(self.img_list)

    def _make_list(self, out_name):image_path = join('./datasets', self.phase + '_image.txt')
        gt_path = join('./datasets', self.phase + '_gt.txt')
        assert exists(image_path)
        
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        
        if exists(gt_path):
            self.gt_list = [line.strip() for line in open(gt_path, 'r')]
            assert len(self.image_list) == len(self.gt_list)
                    self.img_list = [line.strip() for line in open(img_path, 'r')]
