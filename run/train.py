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
    for i, (inputs, gts) in enumerate(tqdm(train_loader, desc="Epoch: {:d} Output: {}".format(epoch, output_path))):
        data_time.update(time.time() - end)

        ## loading image pairs
        inputs = inputs.float().cuda()
        gts = gts.float().cuda()

        ## feed-forward the data into network
        outs = model(inputs)        
        optim.zero_grad()
        dis_optim.zero_grad()
        
        ## Calculate the loss
        ## Base loss
        loss = criterion(outs, gts)
        ## GAN loss
        dis_loss = dis_criterion(gts, target_is_real=True) + dis_criterion(dis_model(outs.detach()), target_is_real=False)
        gen_loss = dis_criterion(dis_model(outs), target_is_real=True)
        ## Total loss
        total_loss = loss+gan_weight*(dis_loss+gen_loss)
        losses.update(total_loss.data, inputs.size(0))
        
        ## backward and update the network
        total_loss.backward()
        optim.step()
        dis_optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        base_losses.append(loss.item())
        gan_losses.append((dis_loss+gen_loss).item())
        total_losses.append(total_loss.item())

    return [iters, total_losses, gan_losses, base_losses]