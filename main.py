import os
import time
import shutil
import sys
import logging
import argparse
import threading
import numpy as np
from os.path import join, exists, split
from math import log10
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torchvision import datasets

import lib.datasets.transforms as transforms
from models.network import MSBDN
from models.loss import LossFunction
from lib.datasets.dataset import RestList
from lib.utils.util import save_output_images, save_checkpoint, psnr, AverageMeter


def train(train_loader, model, optim, criterion, epoch, eval_score=None, print_freq=10, logger=None):
    
    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    
    #######################################
    # (2) Training
    #######################################
    
    for i, (inp, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        ## loading image pairs
        img = torch.autograd.Variable((inp.float()).cuda())
        gt = torch.autograd.Variable((label.float()).cuda())

        ## feed-forward the data into network
        _, out = model(img)        
        optim.zero_grad()
        
        ## calculate the loss
        loss = criterion(out, gt)
        losses.update(loss.data, inp.size(0))

        ## backward and update the network
        loss.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('E : [{0}][{1}/{2}]\t'
                        'T {batch_time.val:.3f}\t'
                        'Loss {s.val:.3f} ({s.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, s=losses))

def validate(val_loader, model, batch_size, crop_size=256, flag = False, eval_score=None, print_freq=10, output_dir='val', \
    save_vis=False, epoch=None, logger=None):

    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    score = AverageMeter()
    model.eval()
    end = time.time()

    #######################################
    # (2) Inference
    #######################################

    for i, (inp, gt, name) in enumerate(val_loader):

        # loading image pairs
        img = (inp.float()).cuda()
        gt = gt.float()

        # template parameters
        # position = []                       # position queue
        # batch_count = 0                     # num
        # _, _, H, W = img.size()             # image size
        # result = torch.zeros(1,3,H,W)       # output image
        # voting_mask = torch.zeros(1,1,H,W)  # denominator

        # # cropping images into 256x256 patches, feed-forwarding to network, and collect
        # for top in range(0, H, 128):
        #     for left in range(0, W, 128):
                                
        #         piece = torch.zeros(1, 3, crop_size, crop_size)
        #         piece = img[:, :, top:top+crop_size, left:left+crop_size] # cropped patches

        #         _, _, h, w = piece.size()
        #         if (h != crop_size) or (w != crop_size) : # non-regular sized patches
        #             # inference the non-regular sized patche first
        #             with torch.no_grad():
        #                 piece = F.interpolate(piece, size=(256, 256), mode='bilinear')
        #                 pred_crop = model(piece)
        #                 pred_crop = F.interpolate(pred_crop, size=(h, w), mode='bilinear')
                    
        #             # assign the result on output image
        #             result[0, :, top:top+crop_size, left:left+crop_size] += pred_crop[0,:,:,:].cpu()
        #             voting_mask[:, :, top:top+crop_size, left:left+crop_size] += 1
                    
        #             # inference the patches in the patch queue
        #             with torch.no_grad():
        #                 pred_crop = model(crop)

        #             # initialize the batch count
        #             batch_count = 0

        #             # assign the results on output image
        #             for num, (t, l) in enumerate(position):
        #                 result[0, :, t:t+crop_size, l:l+crop_size] += pred_crop[num, :, :, :].cpu()
        #                 voting_mask[:, :, t:t+crop_size, l:l+crop_size] += 1
                    
        #             # initialize the position queue
        #             position = []

        #         else : # regular sized patch
                    
        #             if batch_count > 0: # push patch into the patch queue
        #                 crop = torch.cat((crop, piece), dim=0)
        #             else :              # initialize the patch queue
        #                 crop = piece

        #             # push position into position queue
        #             position.append([top, left])
        #             batch_count += 1

        #             # inference the patches in the patch queue
        #             if batch_count == batch_size:
        #                 with torch.no_grad():
        #                     pred_crop = model(crop)
        #                 batch_count = 0
        #                 for num, (t, l) in enumerate(position):
        #                     result[0, :, t:t+crop_size, l:l+crop_size] += pred_crop[num, :, :, :].cpu()
        #                     voting_mask[:, :, t:t+crop_size, l:l+crop_size] += 1

        #                 # initialize the position queue
        #                 position = []

        # # post processing
        # out = result/voting_mask
        # out = torch.clamp(out,min=0, max=1)
        # out = out * 255
        # gt = gt * 255



        # full resolution processing
        _, _, h, w = img.size()
        img = F.interpolate(img, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')

        with torch.no_grad():
            Sat, out = model(img)
        out = torch.clamp(out,min=0, max=1)

        out = F.interpolate(out, size=(h,w), mode='bilinear')
        out = out.cpu()
        out = out * 255
        gt = gt * 255

        # evaluation
        if eval_score is not None:
            logger.info(' {k}th PSNR is {s:.3f}'.format(k=i, s=eval_score(out, gt, 255)))
            score.update(eval_score(out, gt, 255), inp.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if save_vis == True:
            save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch))
            out = out.data.numpy()
            save_output_images(out, '', name, save_dir, epoch)
            mask = Sat.cpu().data.numpy() * 10
            save_output_images(mask, '_mask', name, save_dir, epoch)

    if logger is not None:
        logger.info(' * Score is {s.avg:.3f}'.format(s=score))
    return score.avg

def run(args, saveDirName='.', logger=None):
    #######################################
    # (1) Load and display hyper-parameters
    #######################################

    batch_size = args.batch_size
    crop_size = args.crop_size    
    print(' '.join(sys.argv))
    for k, v in args.__dict__.items():
        logger.info('{0}:\t{1}'.format(k, v))
    
    #######################################
    # (2) Initialize loaders
    #######################################

    data_dir = args.data_dir
    t_super= [transforms.RandomCrop(crop_size),
                transforms.RandomFlip(),
                transforms.ToTensor()]
    train_loader = torch.utils.data.DataLoader(
        RestList(data_dir, 'train', transforms.Compose(t_super)),
        batch_size=batch_size, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=False)

    t = [transforms.ToTensor()]
    val_loader = torch.utils.data.DataLoader(
        RestList(data_dir, 'val', transforms.Compose(t), out_name=True),
        batch_size=1, shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        RestList(data_dir, 'test', transforms.Compose(t), out_name=True),
        batch_size=1, shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)

    #######################################
    # (3) Initialize neural netowrk and optimizer
    #######################################

    model = MSBDN()
    model = torch.nn.DataParallel(model).cuda()
    optim = torch.optim.Adam(model.parameters(),args.lr)

    #######################################
    # (4) Define loss function
    #######################################

    criterion = LossFunction().cuda()

    #######################################
    # (5) Train or test
    #######################################

    cudnn.benchmark = True
    best_prec1 = 0
    lr = args.lr
    if args.cmd == 'train' : # train mode
        for epoch in range(args.epochs):
            logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

            ## train the network
            train(train_loader, model, optim, criterion, epoch, eval_score=psnr, logger=logger)        

            ## validate the network
            val_score = validate(val_loader, model, batch_size=batch_size, output_dir = saveDirName, save_vis=True, epoch=epoch+1, eval_score=psnr, logger=logger)

            ## save the neural network
            if best_prec1 < val_score : 
                best_prec1 = val_score
                # checkpoint for g
                history_path_g = saveDirName + '/' + 'checkpoint_{:03d}_'.format(epoch + 1) + str(best_prec1)[:6] + '.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                }, True, filename=history_path_g)

    else :  # test mode (if epoch = 0, the image format is png)
        checkpoint = torch.load('model_params.tar')
        model.load_state_dict(checkpoint['model'])
        _ = validate(test_loader, model, batch_size=batch_size, crop_size=crop_size, output_dir='test', save_vis=True, epoch=0, eval_score=psnr, logger=logger)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test']) #
    parser.add_argument('--data-dir', default=None, required=True) #
    parser.add_argument('--save-dir', default=None, required=True) #
    parser.add_argument('--crop-size', default=0, type=int) #
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
    timeName = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute)
    saveDirName = os.path.join(args.save_dir, timeName)
    os.makedirs(saveDirName, exist_ok=True)

    # logging configuration
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(saveDirName + '/log_training.log')
    logger.addHandler(file_handler)

    run(args, saveDirName=saveDirName, logger=logger)

if __name__ == '__main__':
    main()



