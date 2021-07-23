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
    for i, (img_up_left, img_up_right, img_down_left, img_down_right, gt, name) in enumerate(tqdm(val_loader, desc="Validation iteration")):
        # loading image pairs
        img_up_left = img_up_left.float().cuda()
        img_up_right = img_up_right.float().cuda()
        img_down_left = img_down_left.float().cuda()
        img_down_right = img_down_right.float().cuda()
        gt = gt.float()
        with torch.no_grad():
            out_up_left = model(img_up_left).cpu()
            out_up_right = model(img_up_right).cpu()
            out_down_left = model(img_down_left).cpu()
            out_down_right = model(img_down_right).cpu()

        # print('out_up_left.shape: ', out_up_left.shape)
        # print('out_up_right.shape: ', out_up_right.shape)
        # print('out_down_left.shape: ', out_down_left.shape)
        # print('out_down_right.shape: ', out_down_right.shape)
        out_up = (torch.cat([out_up_left[:, :, :, 0:gt.shape[3]//2].permute(0, 2, 3, 1), out_up_right[:, :, :, (out_up_right.shape[3]-gt.shape[3]//2):].permute(0, 2, 3, 1)],2)).permute(0, 3, 1, 2)
        out_down = (torch.cat([out_down_left[:, :, :, 0:gt.shape[3]//2].permute(0, 2, 3, 1), out_down_right[:, :, :, (out_down_right.shape[3]-gt.shape[3]//2):].permute(0, 2, 3, 1)],2)).permute(0, 3, 1, 2)
        # print('out_up.shape: ', out_up.shape)
        # print('out_down.shape: ', out_down.shape)
        out = (torch.cat([out_up[:, :, 0:gt.shape[2]//2, :].permute(0, 2, 3, 1), out_down[:, :, (out_down.shape[2]-gt.shape[2]//2):, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)
        # out = (torch.cat([out_up[:, :, 0:1224, :].permute(0, 2, 3, 1), out_down[:, :, 1208:, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)
        # print('out.shape: ', out.shape)
        # print('gt.shape: ', gt.shape)
        assert out.shape[2] == gt.shape[2]
        assert out.shape[3] == gt.shape[3]
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
