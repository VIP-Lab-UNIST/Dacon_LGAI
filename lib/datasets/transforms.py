import numbers
import random

import numpy as np
from PIL import Image, ImageOps
import torch

"""
Modified data transforms for [image, inter, gt] triplet
"""
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        assert label is None or (image.size == label.size), \
            "image and label doesn't have the same size {} / {}".format(
                image.size, label.size)

        w, h = image.size
        tw = self.size
        th = self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        results = [image.crop((x1, y1, x1 + tw, y1 + th))]
        if label is not None:
            results.append(label.crop((x1, y1, x1 + tw, y1 + th)))

        return results
        
class RandomFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.333:
            results = [image.transpose(Image.FLIP_LEFT_RIGHT),
                       label.transpose(Image.FLIP_LEFT_RIGHT)]
        elif random.random() < 0.666:
            results = [image.transpose(Image.FLIP_TOP_BOTTOM),
                       label.transpose(Image.FLIP_TOP_BOTTOM)]
        else:
            results = [image, label]
        return results

class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        
        if label is None:
            return img,
        
        else:
            gt = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes()))
            if label.mode == 'YCbCr':
                nchannel=3
            else:
                nchannel = len(label.mode)

            gt = gt.view(label.size[1], label.size[0], nchannel)
            gt = gt.transpose(0, 1).transpose(0, 2).contiguous()
            gt = gt.float().div(255)
            #return img, torch.LongTensor(np.array(label, dtype=np.int))
            return img, gt

class Compose(object):
    """Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
