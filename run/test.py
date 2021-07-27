import os
import time
import torch
import zipfile
from os.path import basename
from glob import glob
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
        if (name[0] == 'train_input_10330.png') or (name[0] == 'train_input_10363.png') or (name[0] == 'train_input_10570.png'):
            continue

        # loading image pairs
        img = img.float().cuda()
        gt = gt.float()
        # with torch.no_grad():
        #     out = model(img).cpu()

        division = 32 * 2
        value_w = img.shape[2]//division
        remainder_w = img.shape[2]%division
            
        value_h = img.shape[3]//division
        remainder_h = img.shape[3]%division

        img_up_left = img[:, :, 0:(value_w*division)//2, 0:(value_h*division)//2]
        img_up_right = img[:, :, 0:(value_w*division)//2, (img.shape[3]-remainder_h)//2:img.shape[3]]
        img_down_left = img[:, :, (img.shape[2]-remainder_w)//2:img.shape[2], 0:(value_h*division)//2]
        img_down_right = img[:, :, (img.shape[2]-remainder_w)//2:img.shape[2], (img.shape[3]-remainder_h)//2:img.shape[3]]

        # img_up_left = img[:, :, 0:(value_w*division)//2, 0:(value_h*division)//2]
        # img_up_right = img[:, :, 0:(value_w*division)//2, remainder_h+(img.shape[3]-remainder_h)//2:img.shape[3]]
        # img_down_left = img[:, :, remainder_w+(img.shape[2]-remainder_w)//2:img.shape[2], 0:(value_h*division)//2]
        # img_down_right = img[:, :, remainder_w+(img.shape[2]-remainder_w)//2:img.shape[2], remainder_h+(img.shape[3]-remainder_h)//2:img.shape[3]]
        
        assert img_up_left.shape[2]+img_down_left.shape[2] >= gt.shape[2]
        assert img_up_left.shape[3]+img_down_left.shape[3] >= gt.shape[3]

        with torch.no_grad():
            out_up_left = model(img_up_left)[0].cpu()
            out_up_right = model(img_up_right)[0].cpu()
            out_down_left = model(img_down_left)[0].cpu()
            out_down_right = model(img_down_right)[0].cpu()
        
        out_up = (torch.cat([out_up_left[:, :, :, 0:gt.shape[3]//2], out_up_right[:, :, :, (out_up_right.shape[3]-gt.shape[3]//2):]],3))
        out_down = (torch.cat([out_down_left[:, :, :, 0:gt.shape[3]//2], out_down_right[:, :, :, (out_down_right.shape[3]-gt.shape[3]//2):]],3))
        
        out = (torch.cat([out_up, out_down],2))

        # evaluation
        score.update(psnr(out, gt, 1.), out.size(0))

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
        logger.info(' * Score is {s.avg:.3f}'.format(s=score))

    return score.avg