import os
import time
from tqdm import tqdm
from lib.utils.util import AverageMeter
import torch
import numpy as np

def train(train_loader, models, optims, criterions, edge_weight, eval_score=None, print_freq=10, logger=None):
    
    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Model
    model = models
    model.train()
    # Optimizer
    optim = optims
    # Criterions
    criterion_char = criterions[0]
    criterion_edge = criterions[1]

    end = time.time()
    
    #######################################
    # (2) Training
    #######################################
    iters = []
    loss_chares = []
    loss_edges = []
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
        loss_char = np.sum([criterion_char(outs[j],gts) for j in range(len(outs))])
        loss_edge = np.sum([criterion_edge(outs[j],gts) for j in range(len(outs))])
        loss_total = (loss_char) + (edge_weight*loss_edge)

        losses.update(loss_total.data, inputs.size(0))
        
        ## backward and update the network
        loss_total.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iters.append(i)
        loss_chares.append(loss_char.item())
        loss_edges.append(loss_edge.item())
        total_losses.append(loss_total.item())

    return [iters, total_losses, loss_chares, loss_edges]