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
    for i, (img_up, img_down, gt, name) in enumerate(tqdm(val_loader, desc="Validation iteration")):
        # loading image pairs
        img_up = img_up.float().cuda()
        img_down = img_down.float().cuda()
        gt = gt.float()
        if (name[0] == 'train_input_10330.png') or (name[0] == 'train_input_10363.png') or (name[0] == 'train_input_10570.png'):
            continue
        with torch.no_grad():
            out_up, _, _, _, _ = model(img_up)
            out_up = out_up.cpu()
            out_down, _, _, _, _  = model(img_down)
            out_down = out_down.cpu()

        out = (torch.cat([out_up[:, :, 0:1224, :].permute(0, 2, 3, 1), out_down[:, :, 1208:, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)

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
