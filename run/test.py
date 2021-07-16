import os
import time
import torch
from tqdm import tqdm
from lib.utils.util import save_output_images, save_checkpoint, psnr, AverageMeter

def validate(val_loader, model, batch_size, output_dir='val', save_vis=False, epoch=None, logger=None, phase=None):

    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    score = AverageMeter()
    model.eval()
    end = time.time()

    #######################################
    # (2) Inference
    #######################################
    for i, (img, gt, name) in enumerate(tqdm(val_loader, desc="Validation iteration")):
        # loading image pairs
        img = img.float().cuda()
        gt = gt.float()
        with torch.no_grad():
            out = model(img).cpu()

        # evaluation
        score.update(psnr(out, gt, 1.), out.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if save_vis == True:
            save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch))
            save_output_images(out, str(epoch), name, save_dir, epoch, phase)
        
    if logger is not None:
        logger.info('E : [{0}]'.format(epoch))
        logger.info(' * Score is {s.avg:.3f}'.format(s=score))

    return score.avg