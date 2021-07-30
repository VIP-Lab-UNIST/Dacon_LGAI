import numbers
import random
import numpy as np
from PIL import Image
import torch
import cv2

"""
Data augmentation for 2D image restoration/enhancement
All the data augmentation classes are designed for the 
numpy array in the size of (H, W, C)
"""

class Resize(object):
    """
    Resize image(s) to the given size or scale factor 
    Either specific size(tuple or int) or the scale factor
    can be used to resize tensors.
    """
    def __init__(self, size=None, scale_factor=None):
        self.size = size
        self.scale_factor = scale_factor

    def __call__(self, *input):
        if self.size is not None:
            assert self.scale_factor is None
            self.size = self.size[::-1] if isinstance(self.size, tuple) \
                                            else (self.size, self.size)
            input = tuple(map(lambda x: cv2.resize(
                        x, dsize=self.size, interpolation='cv2.INTER_LINEAR'
                ), input))

        elif self.scale_factor is not None:
            assert self.size is None
            input = tuple(map(lambda x: cv2.resize(
                        x, dsize=(0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation='cv2.INTER_LINEAR'
                ), input))
        
        return input


class RandomIdentityMapping(object):
    """
    Randomly set the (input, target) pair as (target, target).
    p is used to set the probability of the identity mapping.
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, *input):
        if random.random() < self.p:
            input = (input[1], input[1])
            
        return input


class RandomCrop(object):
    """
    Randomly crop images to the given size.
    Explicit size in the type of tuple or int is used 
    to crop the tensor in desireable size.
    """
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, *input):
        h, w, _ = input[0].shape
        th, tw = self.size
        top = bottom = left = right = 0

        if w < tw:
            left = (tw - w) // 2
            right = tw - w - left
        if h < th:
            top = (th - h) // 2
            bottom = th - h - top
        if left > 0 or right > 0 or top > 0 or bottom > 0:
            input = tuple(map(lambda x: cv2.copyMakeBorder(
                x, top, bottom, left, right, cv2.BORDER_REFLECT), input))
        
        h, w, _ = input[0].shape
        if h == th and w == tw:
            return input

        y1 = random.randint(0, h - th)
        x1 = random.randint(0, w - tw)
        
        input = tuple(map(lambda x: x[y1:y1+th, x1:x1+tw, :], input))

        return input


class RandomScale(object):
    """
    Scale images in the random ratio 
    in range[1, scale] or [scale1, scale2]
    """
    def __init__(self, scale=(0.5, 2.0)):
        self.scale = scale if isinstance(scale, tuple) else (1, scale)

    def __call__(self, *input):
        ratio = random.uniform(*self.scale)
        if ratio == 1:
            return input
        elif ratio < 1:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        input = tuple(map(lambda x: cv2.resize(x, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=interpolation), input))
        return input


class RandomRotate(object):
    """
    Randomly rotates images in (90*n) degree.
    (n = 0, 1, 2, 3)
    """
    def __call__(self, *input):
        assert input[0].shape == input[1].shape
        p = random.random()
        if p < 0.25:
            pass
        elif p < 0.5:
            input = tuple(map(lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), input))
        elif p < 0.75:
            input = tuple(map(lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE), input))
        else:
            input = tuple(map(lambda x: cv2.rotate(x, cv2.ROTATE_180), input))
        return input

class Random180Rotate(object):
    """
    Randomly rotates images in (180*n) degree.
    (n = 0, 1)
    """
    def __call__(self, *input):
        assert input[0].shape == input[1].shape
        p = random.random()
        if p < 0.5:
            pass
        else:
            input = tuple(map(lambda x: cv2.rotate(x, cv2.ROTATE_180), input))
        return input

class RandomFlip(object):
    """
    Randomly flips images horizontally or vertically
    """
    def __call__(self, *input):
        if random.random() < 0.5:
            input = tuple(map(lambda x: cv2.flip(x, 0), input))
        
        if random.random() < 0.5:
            input = tuple(map(lambda x: cv2.flip(x, 1), input))
        
        return input


class ToTensor(object):
    """
    Converts images in the shape of (H, W, C) in the range [0, 255] 
    to a torch.FloatTensor of shape (C, H, W) in the range [0.0, 1.0].
    """

    def __call__(self, *input):
        input = tuple(map(
            lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous().float().div(255.0), input
            ))

        return input


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
