import os
import time
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
    Gen, Dis = models
    Gen.train()
    Dis.train()

    # Optimizer
    optim_Gen, optim_Dis = optims

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

        ## feed-forward the data into network
        outs = Gen(inputs)        
        optim_Gen.zero_grad()
        optim_Dis.zero_grad()
        
        ## Calculate the loss
        ## Base loss
        loss = criterion(outs, gts)
        ## GAN loss
        loss_Dis = gan_criterion(gts, target_is_real=True) + gan_criterion(Dis(outs.detach()), target_is_real=False)
        loss_Gen = gan_criterion(Dis(outs), target_is_real=True)
        ## Total loss
        total_loss = loss+gan_weight*(loss_Dis+loss_Gen)
        losses.update(total_loss.data, inputs.size(0))
        
        ## backward and update the network
        total_loss.backward()
        optim_Gen.step()
        optim_Dis.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        base_losses.append(loss.item())
        gan_losses.append((loss_Dis+loss_Gen).item())
        total_losses.append(total_loss.item())

    return [iters, total_losses, gan_losses, base_losses]