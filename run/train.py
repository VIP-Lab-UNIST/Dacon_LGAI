import os
import time
import torch.nn.functional as F
from tqdm import tqdm
from lib.utils.util import AverageMeter

def train(train_loader, models, optims, criterions, epoch, output_path, gan_weight, eval_score=None, print_freq=10, logger=None):
    
    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Model
    Gen, Dis, Dis1, Dis2, Dis3, Dis4  = models
    Gen.train()
    Dis.train()
    Dis1.train()
    Dis2.train()
    Dis3.train()
    Dis4.train()

    # Optimizer
    optim_Gen, optim_Dis, optim_Dis1, optim_Dis2, optim_Dis3, optim_Dis4 = optims

    # Criterions
    criterion, gan_criterion = criterions

    end = time.time()
    #######################################
    # (2) Training
    #######################################
    iters = []
    base_losses = []
    gan_losses = []
    total_losses = []
    for i, (inputs, gts) in enumerate(tqdm(train_loader, desc="Epoch: {:d} Output: {}".format(epoch, output_path))):
        data_time.update(time.time() - end)

        ## loading image pairs
        inputs = inputs.float().cuda()
        gts = gts.float().cuda()
        gts2 = F.interpolate(gts, scale_factor=0.5)
        gts3 = F.interpolate(gts2, scale_factor=0.5)
        gts4 = F.interpolate(gts3, scale_factor=0.5)

        ## feed-forward the data into network
        outs, outs1, outs2, outs3, outs4 = Gen(inputs)      
        optim_Gen.zero_grad()
        optim_Dis.zero_grad()
        optim_Dis1.zero_grad()
        optim_Dis2.zero_grad()
        optim_Dis3.zero_grad()
        optim_Dis4.zero_grad()
        
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
        dis_loss = gan_criterion(gts, target_is_real=True) + gan_criterion(Dis(outs.detach()), target_is_real=False)
        dis_loss1 = gan_criterion(gts, target_is_real=True) + gan_criterion(Dis1(outs1[0].detach()), target_is_real=False) + \
                    gan_criterion(gts, target_is_real=True) + gan_criterion(Dis1(outs1[1].detach()), target_is_real=False)

        dis_loss2 = gan_criterion(gts, target_is_real=True) + gan_criterion(Dis2(outs2[0].detach()), target_is_real=False) + \
                    gan_criterion(gts, target_is_real=True) + gan_criterion(Dis2(outs2[1].detach()), target_is_real=False)

        dis_loss3 = gan_criterion(gts, target_is_real=True) + gan_criterion(Dis3(outs3[0].detach()), target_is_real=False) + \
                    gan_criterion(gts, target_is_real=True) + gan_criterion(Dis3(outs3[1].detach()), target_is_real=False)

        dis_loss4 = gan_criterion(gts, target_is_real=True) + gan_criterion(Dis4(outs4[0].detach()), target_is_real=False) + \
                    gan_criterion(gts, target_is_real=True) + gan_criterion(Dis4(outs4[1].detach()), target_is_real=False)
        
        gen_loss = gan_criterion(Dis(outs), target_is_real=True)
        gen_loss1 = gan_criterion(Dis1(outs1[0]), target_is_real=True) + \
                    gan_criterion(Dis1(outs1[1]), target_is_real=True)

        gen_loss2 = gan_criterion(Dis2(outs2[0]), target_is_real=True) + \
                    gan_criterion(Dis2(outs2[1]), target_is_real=True)

        gen_loss3 = gan_criterion(Dis3(outs3[0]), target_is_real=True) + \
                    gan_criterion(Dis3(outs3[1]), target_is_real=True)

        gen_loss4 = gan_criterion(Dis4(outs4[0]), target_is_real=True) + \
                    gan_criterion(Dis4(outs4[1]), target_is_real=True)
        
        gan_loss = (dis_loss+dis_loss1+dis_loss2+dis_loss3+dis_loss4 + gen_loss+gen_loss1+gen_loss2+gen_loss3+gen_loss4)
        ## Total loss
        total_loss = base_loss+gan_weight*(gan_loss)
        losses.update(total_loss.data, inputs.size(0))
        
        ## backward and update the network
        total_loss.backward()
        optim_Gen.step()
        optim_Dis.step()
        optim_Dis1.step()
        optim_Dis2.step()
        optim_Dis3.step()
        optim_Dis4.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        base_losses.append(loss.item())
        gan_losses.append(gan_loss.item())
        total_losses.append(total_loss.item())

    return [iters, total_losses, gan_losses, base_losses]

