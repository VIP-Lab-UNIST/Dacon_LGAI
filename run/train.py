import os
import time
import torch.nn.functional as F
from tqdm import tqdm
from lib.utils.util import AverageMeter

def train(train_loader, models, optims, criterions, gan_weight, eval_score=None, print_freq=10, logger=None):
    
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
    base_losses = []
    gan_losses = []
    total_losses = []
    for i, (inputs, gts) in enumerate(tqdm(train_loader, desc="Training iteration")):
        data_time.update(time.time() - end)

        ## loading image pairs
        inputs = inputs.float().cuda()
        gts = gts.float().cuda()
        gts2 = F.interpolate(gts, scale_factor=0.5)
        gts3 = F.interpolate(gts2, scale_factor=0.5)
        gts4 = F.interpolate(gts3, scale_factor=0.5)

        ## feed-forward the data into network
        outs, outs1, outs2, outs3, outs4 = model(inputs)      
        optim.zero_grad()
        dis_optim.zero_grad()
        
        ## Calculate the loss
        ## Base loss
        loss = criterion(outs, gts)
        DWT_loss1 = criterion(outs1[0], gts)
        KA_loss1 = criterion(outs1[1], gts)

        DWT_loss2 = criterion(outs2[0], gts2)
        KA_loss2 = criterion(outs2[1], gts2)

        DWT_loss3 = criterion(outs3[0], gts3)
        KA_loss3 = criterion(outs3[1], gts3)

        DWT_loss4 = criterion(outs4[0], gts4)
        KA_loss4 = criterion(outs4[1], gts4)

        base_loss = loss + (DWT_loss1+KA_loss1) + (DWT_loss2+KA_loss2) + (DWT_loss3+KA_loss3) + (DWT_loss4+KA_loss4)
        
        ## GAN loss
        dis_loss = dis_criterion(gts, target_is_real=True) + dis_criterion(dis_model(outs.detach()), target_is_real=False)
        gen_loss = dis_criterion(dis_model(outs), target_is_real=True)
        
        gan_loss = (dis_loss+gen_loss)
        ## Total loss
        total_loss = base_loss+gan_weight*(gan_loss)
        losses.update(total_loss.data, inputs.size(0))
        
        ## backward and update the network
        total_loss.backward()
        optim.step()
        dis_optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        base_losses.append(loss.item())
        gan_losses.append(gan_loss.item())
        total_losses.append(total_loss.item())

    return [iters, total_losses, gan_losses, base_losses]