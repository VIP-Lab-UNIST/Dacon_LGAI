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
                transforms.Random180Rotate(),
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

    Gen = fusion_net()
    Gen = torch.nn.DataParallel(Gen).cuda()
    optim_Gen = torch.optim.Adam(Gen.parameters(), args.lr)
    scheduler_Gen = optim.lr_scheduler.MultiStepLR(optim_Gen, milestones=[2000, 3000, 4000], gamma=0.7)

    Dis = Discriminator()
    Dis1 = Discriminator()
    Dis2 = Discriminator()
    Dis3 = Discriminator()
    Dis4 = Discriminator()
    Dis = torch.nn.DataParallel(Dis).cuda()
    Dis1 = torch.nn.DataParallel(Dis1).cuda()
    Dis2 = torch.nn.DataParallel(Dis2).cuda()
    Dis3 = torch.nn.DataParallel(Dis3).cuda()
    Dis4 = torch.nn.DataParallel(Dis4).cuda()
    optim_Dis = torch.optim.Adam(Dis.parameters(), args.lr)
    optim1_Dis = torch.optim.Adam(Dis1.parameters(), args.lr)
    optim2_Dis = torch.optim.Adam(Dis2.parameters(), args.lr)
    optim3_Dis = torch.optim.Adam(Dis3.parameters(), args.lr)
    optim4_Dis = torch.optim.Adam(Dis4.parameters(), args.lr)

    scheduler_Dis = optim.lr_scheduler.MultiStepLR(optim_Dis, milestones=[2000, 3000, 4000], gamma=0.7)
    scheduler1_Dis = optim.lr_scheduler.MultiStepLR(optim1_Dis, milestones=[2000, 3000, 4000], gamma=0.7)
    scheduler2_Dis = optim.lr_scheduler.MultiStepLR(optim2_Dis, milestones=[2000, 3000, 4000], gamma=0.7)
    scheduler3_Dis = optim.lr_scheduler.MultiStepLR(optim3_Dis, milestones=[2000, 3000, 4000], gamma=0.7)
    scheduler4_Dis = optim.lr_scheduler.MultiStepLR(optim4_Dis, milestones=[2000, 3000, 4000], gamma=0.7)
    
    if args.resume is not None:
        state = torch.load(args.resume)
        start_epoch = state['epoch']
        Gen.load_state_dict(state['Gen'])
        optim_Gen.load_state_dict(state['optim_Gen'])
        scheduler_Gen.load_state_dict(state['scheduler_Gen'])

        Dis.load_state_dict(state['Dis'])
        Dis1.load_state_dict(state['Dis1'])
        Dis2.load_state_dict(state['Dis2'])
        Dis3.load_state_dict(state['Dis3'])
        Dis4.load_state_dict(state['Dis4'])
        optim_Dis.load_state_dict(state['optim_Dis'])
        optim1_Dis.load_state_dict(state['optim1_Dis'])
        optim2_Dis.load_state_dict(state['optim2_Dis'])
        optim3_Dis.load_state_dict(state['optim3_Dis'])
        optim4_Dis.load_state_dict(state['optim4_Dis'])
        scheduler_Dis.load_state_dict(state['scheduler_Dis'])
        scheduler1_Dis.load_state_dict(state['scheduler1_Dis'])
        scheduler2_Dis.load_state_dict(state['scheduler2_Dis'])
        scheduler3_Dis.load_state_dict(state['scheduler3_Dis'])
        scheduler4_Dis.load_state_dict(state['scheduler4_Dis'])
        logger.info('Complete the resume')
    else:
        start_epoch = 0

    #######################################
    # (4) Define loss function
    #######################################

    criterion = LossFunction(weight_ssim=args.ssim_weight, weight_perc=args.perc_weight).cuda()
    gan_criterion = GANLoss().cuda()

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
    models = Gen, Dis, Dis1, Dis2, Dis3, Dis4
    optims = optim_Gen, optim_Dis, optim1_Dis, optim2_Dis, optim3_Dis, optim4_Dis
    criterions = criterion, gan_criterion
    if args.cmd == 'train' : # train mode
        for epoch in range(start_epoch, args.epochs):
            # logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
            ## train the network
            train_losses = train(train_loader, models, optims, criterions, epoch, saveDirName, args.gan_weight, eval_score=psnr, logger=logger)        
            
            scheduler_Gen.step()
            scheduler_Dis.step()

            if epoch%args.save_interval == 0:
                ## validate the network
                val_score = validate(val_loader, Gen, batch_size=batch_size, output_dir = saveDirName, save_vis=True, epoch=epoch, logger=logger, phase='val')

                ## save the neural network
                history_path_g = os.path.join(saveDirName, 'checkpoint_{:03d}'.format(epoch)+'.tar')
                save_checkpoint({
                'epoch': epoch + 1,
                'Gen': Gen.state_dict(),
                'Dis': Dis.state_dict(),
                'Dis1': Dis1.state_dict(),
                'Dis2': Dis2.state_dict(),
                'Dis3': Dis3.state_dict(),
                'Dis4': Dis4.state_dict(),
                'optim_Gen': optim_Gen.state_dict(),
                'optim_Dis': optim_Dis.state_dict(),
                'optim1_Dis': optim1_Dis.state_dict(),
                'optim2_Dis': optim2_Dis.state_dict(),
                'optim3_Dis': optim3_Dis.state_dict(),
                'optim4_Dis': optim4_Dis.state_dict(),
                'scheduler_Gen': scheduler_Gen.state_dict(),
                'scheduler_Dis': scheduler_Dis.state_dict(),
                'scheduler1_Dis': scheduler1_Dis.state_dict(),
                'scheduler2_Dis': scheduler2_Dis.state_dict(),
                'scheduler3_Dis': scheduler3_Dis.state_dict(),
                'scheduler4_Dis': scheduler4_Dis.state_dict(),
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
        val_score = validate(test_loader, Gen, batch_size=batch_size, output_dir=saveDirName, save_vis=True, epoch=start_epoch, logger=logger, phase='test')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test']) #
    parser.add_argument('--data-dir', default=None, required=True) #
    parser.add_argument('--save-dir', default=None, required=True) #
    parser.add_argument('--crop-size', nargs='+', type=int) #
    parser.add_argument('--step', type=int, default=200) #
    parser.add_argument('--save-interval', type=int, default=2) #
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



