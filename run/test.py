import os
import time
import torch
import zipfile
from os.path import basename
from glob import glob
from tqdm import tqdm
from lib.utils.util import save_output_images, save_checkpoint, psnr, AverageMeter
import torch.nn.functional as F

def validate(val_loader, model, batch_size, output_dir='val', save_vis=False, epoch=None, logger=None, phase=None):

    #######################################
    # (1) Initialize    
    #######################################

    batch_time = AverageMeter()
    score = AverageMeter()
    score_img = AverageMeter()
    model.eval()
    end = time.time()

    #######################################
    # (2) Inference
    #######################################
    for i, (img, gt, name) in enumerate(tqdm(val_loader, desc="Validation iteration")):
        
        # loading image pairs
        
        _, _, h, w = img.size()
        img = F.interpolate(img, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        img = img.float().cuda()
        gt = gt.float()
        with torch.no_grad():
            out = model(img).cpu()
        out = F.interpolate(out, size=(h, w), mode='bilinear')
        img = F.interpolate(img, size=(h, w), mode='bilinear')

        # evaluation
        score.update(psnr(out, gt, 1.), out.size(0))
        score_img.update(psnr(img.cpu(), gt, 1.), img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if save_vis == True:
            save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch))
            save_output_images(out, str(epoch), name, save_dir, epoch, phase)
    
    if phase == 'test':
        files = glob(os.path.join(save_dir, '*.png'), recursive=True)
        zip_file = os.path.join(save_dir, 'output.zip')
        fantasy_zip = zipfile.ZipFile(zip_file, 'w')
        for img in files:
            fantasy_zip.write(img, basename(img).replace('_input', ''))
        fantasy_zip.close()

    if logger is not None:
        logger.info('E : [{0}]'.format(epoch))
        logger.info(' * Score is {s.avg:.3f}/{s_img.avg:.3f}, Time is {batch_time.val:.3f}'.format(s=score, s_img=score_img, batch_time=batch_time))

    return score.avg