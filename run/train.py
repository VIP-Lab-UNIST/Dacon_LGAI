import os
import time
from tqdm import tqdm
from lib.utils.util import AverageMeter

def train(train_loader, models, optims, criterions, eval_score=None, print_freq=10, logger=None):
    
    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Model
    model = models
    model.train()
    optim = optims
    criterion = criterions

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

        ## feed-forward the data into network
        outs = model(inputs)        
        optim.zero_grad()
        
        ## Calculate the loss
        loss = criterion(inputs, outs, gts)
        losses.update(loss.data, inputs.size(0))
        
        ## backward and update the network
        loss.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        total_losses.append(loss.item())

    return [iters, total_losses]