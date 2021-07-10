import os
import time
import shutil
import sys
import logging
import argparse
import threading
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from models.network import Net, Discriminator
from models.loss import LossFunction, GANLoss
from lib.datasets.dataset import RestList
from lib.utils.util import save_output_images, save_checkpoint, psnr, AverageMeter


def train(train_loader, models, optims, criterions, epoch, eval_score=None, print_freq=10, logger=None, path=None):
    
    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Model
    model = models[0]
    dis_model = models[1]
    model.train()
    dis_model.train()
    # Optimizer
    optim = optims[0]
    dis_optim = optims[1]
    # Criterions
    criterion = criterions[0]
    dis_criterion = criterions[1]

    end = time.time()
    
    #######################################
    # (2) Training
    #######################################
    iters = []
    basic_losses = []
    gan_losses = []
    total_losses = []
    for i, (inp, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        ## loading image pairs
        img = torch.autograd.Variable((inp.float()).cuda())
        gt = torch.autograd.Variable((label.float()).cuda())

        ## feed-forward the data into network
        out = model(img)        
        optim.zero_grad()
        
        ## calculate the loss
        ## Basic loss
        loss = criterion(out, gt)
        ## GAN loss
        # Discriminator loss
        dis_loss = dis_criterion(gt, target_is_real=True) + dis_criterion(dis_model(out), target_is_real=False)
        # Generator loss
        gen_loss = dis_criterion(dis_model(out), target_is_real=True)
        ## Total loss
        total_loss = loss+0.5*(dis_loss+gen_loss)
        losses.update(total_loss.data, inp.size(0))
        
        iters.append(i)
        basic_losses.append(loss.item())
        gan_losses.append((dis_loss+gen_loss).item())
        total_losses.append(total_loss.item())

        ## backward and update the network
        total_loss.backward()
        optim.step()
        dis_optim.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            logger.info('E : [{0}][{1}/{2}]'.format(epoch, i, len(train_loader)))
    
    ## save and plot the loss
    if path is not None:
        plt.plot(iters, total_losses, 'r', label='Total loss')
        plt.plot(iters, gan_losses, 'g', label='GAN loss')
        plt.plot(iters, basic_losses, 'b', label='Base loss')
        plt.legend(loc='upper right')   
        plt.xlabel('Iterations')
        plt.ylabel('Losses')
        plt.grid()
        plt.savefig(join(path, 'losses_%depoch.png'%epoch))
        plt.clf()
        plt.cla()
    
def validate(val_loader, model, batch_size, output_dir='val', save_vis=False, epoch=None, logger=None):

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

        with torch.no_grad():
            out = model(img).cpu()

        # evaluation
        score.update(psnr(out, gt, 255), inp.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if save_vis == True:
            save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch))
            out = out.data.numpy()
            save_output_images(out, str(epoch), name, save_dir, epoch)

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

    model = Net()
    model = torch.nn.DataParallel(model).cuda()
    optim = torch.optim.Adam(model.parameters(),args.lr)

    dis = Discriminator()
    dis = torch.nn.DataParallel(dis).cuda()
    dis_optim = torch.optim.Adam(dis.parameters(),args.lr)

    #######################################
    # (4) Define loss function
    #######################################

    criterion = LossFunction().cuda()
    dis_criterion = GANLoss().cuda()

    #######################################
    # (5) Train or test
    #######################################

    cudnn.benchmark = True
    best_prec1 = 0
    lr = args.lr
    plot_val_scores = []
    plot_epochs = []
    if args.cmd == 'train' : # train mode
        for epoch in range(args.epochs):
            logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

            ## train the network
            train(train_loader, [model, dis], [optim, dis_optim], [criterion,dis_criterion], epoch, eval_score=psnr, logger=logger, path=saveDirName)        
            ## validate the network
            val_score = validate(val_loader, model, batch_size=batch_size, output_dir = saveDirName, save_vis=True, epoch=epoch+1, logger=logger)

            ## save the neural network
            if best_prec1 < val_score : 
                best_prec1 = val_score
                # checkpoint for g
                # history_path_g = saveDirName + '/' + 'checkpoint_{:03d}_'.format(epoch + 1) + str(best_prec1)[:6] + '.tar'
                history_path_g = join(saveDirName, 'checkpoint_{:03d}'.format(epoch + 1)+'.tar')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                }, True, filename=history_path_g)

            ## save and plot the valid score
            plot_epochs.append(epoch+1)
            plot_val_scores.append(val_score.item())
            plt.plot(plot_epochs, plot_val_scores, 'r')
            # plt.xticks(np.arange(0, 50,step=2))
            plt.xlabel('Epochs')
            plt.yticks(np.arange(15,35,step=2))
            plt.ylabel('Validation scores')
            plt.grid()
            plt.savefig(join(saveDirName, 'scores.png'))
            plt.clf()
            plt.cla()
            with open(join(saveDirName, 'scores.json'), 'w') as fp:
                json.dump([plot_epochs, plot_val_scores], fp)

    else :  # test mode (if epoch = 0, the image format is png)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        _ = validate(test_loader, model, batch_size=batch_size, output_dir=saveDirName, save_vis=True, epoch=epoch, logger=logger)

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test']) #
    parser.add_argument('--data-dir', default=None, required=True) #
    parser.add_argument('--save-dir', default=None, required=True) #
    parser.add_argument('--crop-size', default=0, type=int) #
    parser.add_argument('--step', type=int, default=200) #
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', #
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



