import os
import time
import torch
from tqdm import tqdm
from lib.utils.util import AverageMeter

def train(train_loader, model, optim, criterion, epoch, output_path, eval_score=None, print_freq=50, logger=None):
    
    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Model
    model.train()
    optim = optim
    criterion = criterion
    end = time.time()
    
    #######################################
    # (2) Training
    #######################################
    iters = []
    total_losses = []
    for i, (inputs, gts) in enumerate(tqdm(train_loader, desc="Epoch: {:d} Output: {}".format(epoch, output_path))):
        data_time.update(time.time() - end)

        ## loading image pairs
        inputs = inputs.float().cuda()
        gts = gts.float().cuda()

        ## feed-forward the data into network
        outs = model(inputs)     
        optim.zero_grad()
        
        ## Calculate the loss
        loss = torch.zeros([1]).cuda()
        for out in outs:
            loss += criterion(out, gts)
        losses.update(loss.item(), inputs.size(0))
        
        ## backward and update the network
        loss.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        total_losses.append(loss.item())

        # if i % print_freq == 0:
        #     logger.info('I: [{now_iter:d}/{total_iter:d}]\t'
        #                 'T: {batch_time.val:.3f}\t'
        #                 'Loss: {s.val:.3f} ({s.avg:.3f})\t'.format(now_iter=i, total_iter=len(train_loader), batch_time=batch_time, s=losses))

    return iters, total_losses