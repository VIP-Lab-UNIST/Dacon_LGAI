import os
import sys
import logging
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import lib.datasets.transforms as transforms
from run.train import train
from run.test import validate
from lib.datasets.dataset import RestList
from models.network import MSBDN, Discriminator
from models.loss import LossFunction, GANLoss
from models.optimizer import CosineAnnealingWarmUpRestarts
from lib.utils.util import save_output_images, save_checkpoint, psnr, plot_losses, plot_scores, plot_lrs

def run(args, saveDirName='.', logger=None):
    #######################################
    # (1) Load and Display hyper-parameters
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
    t_super= [transforms.RandomCrop(tuple(args.crop_size)),
                transforms.RandomFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor()]
    
    train_loader = torch.utils.data.DataLoader(
        RestList(data_dir, 'train', transforms.Compose(t_super)),
        batch_size=batch_size, shuffle=True, num_workers=8,
        pin_memory=False, drop_last=False)

    t = [transforms.ToTensor()]
    val_loader = torch.utils.data.DataLoader(
        RestList(data_dir, 'val', transforms.Compose(t), out_name=True),
        batch_size=1, shuffle=False, num_workers=8,
        pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        RestList(data_dir, 'test', transforms.Compose(t), out_name=True),
        batch_size=1, shuffle=False, num_workers=8,
        pin_memory=False, drop_last=False)

    #######################################
    # (3) Initialize neural netowrk and optimizer
    #######################################

    Gen = MSBDN()
    Gen = torch.nn.DataParallel(Gen).cuda()
    optim_Gen = torch.optim.Adam(Gen.parameters(), args.lr)
    scheduler_Gen = optim.lr_scheduler.MultiStepLR(optim_Gen, milestones=[2000, 3000, 4000], gamma=0.7)
    # scheduler_Gen = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_Gen, T_0=20, T_mult=1, eta_min=0.00001)

    # Dis = Discriminator()
    # Dis = torch.nn.DataParallel(Dis).cuda()
    # optim_Dis = torch.optim.Adam(Dis.parameters(), args.lr)
    # scheduler_Dis = optim.lr_scheduler.MultiStepLR(optim_Dis, milestones=[20, 30, 40], gamma=0.7)
    # scheduler_Dis = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_Dis, T_0=20, T_mult=1, eta_min=0.00001)
    
    if args.resume is not None:
        state = torch.load(args.resume)
        Gen.load_state_dict(state['Gen'])
        optim_Gen.load_state_dict(state['optim_Gen'])
        scheduler_Gen.load_state_dict(state['scheduler_Gen'])

        # Dis.load_state_dict(state['Dis'])
        # optim_Dis.load_state_dict(state['optim_Dis'])
        # scheduler_Dis.load_state_dict(state['scheduler_Dis'])
        start_epoch = state['epoch']
        logger.info('Complete the resume!')
    else:
        start_epoch = 0

    #######################################
    # (4) Define loss function
    #######################################

    criterionPix = LossFunction(ssim_weight=args.ssim_weight, perc_weight=args.perc_weight).cuda()
    # criterionGAN = GANLoss().cuda()

    #######################################
    # (5) Train or test
    #######################################

    plot_val_scores = []
    plot_epochs = []
    plot_iters =  []
    plot_pix_losses= []
    # plot_gan_losses= []
    plot_total_losses= []
    plot_lrs_Dis = []
    plot_lrs_Gen = []

    # models = Gen, Dis
    # optims = optim_Gen, optim_Dis
    # criterions = criterionPix, criterionGAN
    
    models = Gen
    optims = optim_Gen
    criterions = criterionPix
    if args.cmd == 'train' : # train mode
        for epoch in range(start_epoch, args.epochs):
            logger.info('Epoch: [{0}]\t Gen lr {1:.06f}'.format(epoch, optim_Gen.param_groups[0]['lr']))
            # logger.info('Epoch: [{0}]\t Gen lr {1:.06f}\t Dis lr {1:.06f}'.format(epoch, optim_Gen.param_groups[0]['lr'], optim_Dis.param_groups[0]['lr']))
            ## train the network
            train_losses = train(train_loader, models, optims, criterions, epoch, saveDirName, args.gan_weight, eval_score=psnr, logger=logger)        

            ## step the scheduler
            scheduler_Gen.step()
            # scheduler_Dis.step()

            if epoch%80 == 0:

                ## validate the network
                val_score = validate(val_loader, Gen, batch_size=batch_size, output_dir = saveDirName, save_vis=True, epoch=epoch, logger=logger, phase='val')

                ## save the neural network
                history_path_g = os.path.join(saveDirName, 'checkpoint_{:03d}'.format(epoch)+'.tar')
                save_checkpoint({
                    'epoch': epoch,
                    'Gen': Gen.state_dict(),
                    'optim_Gen': optim_Gen.state_dict(),
                    'scheduler_Gen': scheduler_Gen.state_dict(),
                }, True, filename=history_path_g)

                
                #######################################
                # (6) Plotting
                #######################################
                plot_iters.extend(list(map(lambda x: epoch*len(train_loader)+x, train_losses[0])))
                plot_total_losses.extend(train_losses[1])
                plot_pix_losses.extend(train_losses[2])
                plot_epochs.append(epoch)
                plot_val_scores.append(val_score.item())
                
                plot_lrs_Gen.append(optim_Gen.param_groups[0]['lr'])

                ## Loss
                plot_losses(plot_iters, [plot_total_losses, plot_pix_losses], os.path.join(saveDirName, 'losses.jpg'))

                ## Scores
                plot_scores(plot_epochs, plot_val_scores, os.path.join(saveDirName, 'scores.jpg'))
                
                ## Learning rate
                plot_lrs(plot_epochs, [plot_lrs_Gen], os.path.join(saveDirName, 'lrs.jpg'))

    else :  
        val_score = validate(test_loader, Gen, batch_size=batch_size, output_dir=saveDirName, save_vis=True, epoch=start_epoch, logger=logger, phase='test')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test']) #
    parser.add_argument('--data-dir', default=None, required=True) #
    parser.add_argument('--save-dir', default=None, required=True) #
    parser.add_argument('--crop-size', nargs='+', type=int) #
    parser.add_argument('--step', type=int, default=200) #
    parser.add_argument('--ssim_weight', type=float, default=0) #
    parser.add_argument('--perc_weight', type=float, default=0) #
    parser.add_argument('--gan_weight', type=float, default=0) #
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', #
                        help='input batch size for training (default: 64)') #
    parser.add_argument('--epochs', type=int, default=10, metavar='N', #
                        help='number of epochs to train (default: 10)') #
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)') #
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', #
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


