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
from models.network import fusion_net, Discriminator
from models.loss import LossFunction, GANLoss
from lib.utils.util import save_output_images, save_checkpoint, psnr, plot_losses, plot_scores

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
    
    t_super= [transforms.RandomCrop(tuple(args.crop_size)),
                transforms.RandomFlip(),
                transforms.Random180Rotate(),
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

    gen = fusion_net()
    gen = torch.nn.DataParallel(gen).cuda()
    gen_optim = torch.optim.Adam(gen.parameters(), args.lr)
    gen_scheduler = optim.lr_scheduler.MultiStepLR(gen_optim, milestones=[2000, 3000, 4000], gamma=0.7)

    dis = Discriminator()
    dis = torch.nn.DataParallel(dis).cuda()
    dis_optim = torch.optim.Adam(dis.parameters(), args.lr)
    dis_scheduler = optim.lr_scheduler.MultiStepLR(dis_optim, milestones=[2000, 3000, 4000], gamma=0.7)
    
    if args.resume is not None:
        state = torch.load(args.resume)
        start_epoch = state['epoch']
        gen.load_state_dict(state['gen'])
        gen_optim.load_state_dict(state['gen_optim'])
        gen_scheduler.load_state_dict(state['gen_scheduler'])

        dis.load_state_dict(state['dis'])
        dis_optim.load_state_dict(state['dis_optim'])
        dis_scheduler.load_state_dict(state['dis_scheduler'])
        print('Complete the resume')
    else:
        start_epoch = 0

    #######################################
    # (4) Define loss function
    #######################################

    criterion = LossFunction(weight_ssim=args.ssim_weight, weight_perc=args.perc_weight).cuda()
    dis_criterion = GANLoss().cuda()

    #######################################
    # (5) Train or test
    #######################################

    cudnn.benchmark = True
    best_prec1 = 0
    lr = args.lr
    plot_val_scores = []
    plot_epochs = []
    plot_iters =  []
    plot_base_losses= []
    plot_gan_losses= []
    plot_total_losses= []
    if args.cmd == 'train' : # train mode
        for epoch in range(start_epoch, args.epochs):
            # logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
            ## train the network
            train_losses = train(train_loader, [gen, dis], [gen_optim, dis_optim], [criterion,dis_criterion], args.gan_weight, eval_score=psnr, logger=logger)        
            
            gen_scheduler.step()
            dis_scheduler.step()

            if epoch%80 == 0:
                ## validate the network
                val_score = validate(val_loader, gen, batch_size=batch_size, output_dir = saveDirName, save_vis=True, epoch=epoch, logger=logger, phase='val')

                ## save the neural network
                history_path_g = os.path.join(saveDirName, 'checkpoint_{:03d}'.format(epoch)+'.tar')
                save_checkpoint({
                    'epoch': epoch,
                    'gen': gen.state_dict(),
                    'dis': dis.state_dict(),
                    'gen_optim': gen_optim.state_dict(),
                    'dis_optim': dis_optim.state_dict(),
                    'gen_scheduler': gen_scheduler.state_dict(),
                    'dis_scheduler': dis_scheduler.state_dict(),
                }, True, filename=history_path_g)

                #######################################
                # (6) Plotting
                #######################################
                plot_iters.extend(list(map(lambda x: epoch*len(train_loader)+x, train_losses[0])))
                plot_total_losses.extend(train_losses[1])
                plot_gan_losses.extend(train_losses[2])
                plot_base_losses.extend(train_losses[3])
                plot_epochs.append(epoch)
                plot_val_scores.append(val_score.item())
                ## Loss
                plot_losses(plot_iters, [plot_total_losses, plot_base_losses, plot_gan_losses], os.path.join(saveDirName, 'losses.jpg'))

                ## Scores
                plot_scores(plot_epochs, plot_val_scores, os.path.join(saveDirName, 'scores.jpg'))

    else :  # test mode (if epoch = 0, the image format is png)
        val_score = validate(test_loader, gen, batch_size=batch_size, output_dir=saveDirName, save_vis=True, epoch=start_epoch, logger=logger, phase='test')

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



