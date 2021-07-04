def validate(val_loader, model, batch_size, crop_size=256, flag = False, eval_score=None, print_freq=10, output_dir='val', \
    save_vis=False, epoch=None, logger=None):

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

    for i, (inp, gt, name) in enumerate(val_loader):

        # loading image pairs
        img = (inp.float()).cuda()
        gt = gt.float()

        # template parameters
        position = []                       # position queue
        batch_count = 0                     # num
        _, _, H, W = img.size()             # image size
        result = torch.zeros(1,3,H,W)       # output image
        voting_mask = torch.zeros(1,1,H,W)  # denominator

        # cropping images into 256x256 patches, feed-forwarding to network, and collect
        for top in range(0, H, 128):
            for left in range(0, W, 128):
                                
                piece = torch.zeros(1, 3, crop_size, crop_size)
                piece = img[:, :, top:top+crop_size, left:left+crop_size] # cropped patches

                _, _, h, w = piece.size()
                if (h != crop_size) or (w != crop_size) : # non-regular sized patches
                    # inference the non-regular sized patche first
                    with torch.no_grad():
                        pred_crop = model(piece)
                    
                    # assign the result on output image
                    result[0, :, top:top+crop_size, left:left+crop_size] += pred_crop[0,:,:,:].cpu()
                    voting_mask[:, :, top:top+crop_size, left:left+crop_size] += 1
                    
                    # inference the patches in the patch queue
                    with torch.no_grad():
                        pred_crop = model(crop)

                    # initialize the batch count
                    batch_count = 0

                    # assign the results on output image
                    for num, (t, l) in enumerate(position):
                        result[0, :, t:t+crop_size, l:l+crop_size] += pred_crop[num, :, :, :].cpu()
                        voting_mask[:, :, t:t+crop_size, l:l+crop_size] += 1
                    
                    # initialize the position queue
                    position = []

                else : # regular sized patch
                    
                    if batch_count > 0: # push patch into the patch queue
                        crop = torch.cat((crop, piece), dim=0)
                    else :              # initialize the patch queue
                        crop = piece

                    # push position into position queue
                    position.append([top, left])
                    batch_count += 1

                    # inference the patches in the patch queue
                    if batch_count == batch_size:
                        with torch.no_grad():
                            pred_crop = model(crop)
                        batch_count = 0
                        for num, (t, l) in enumerate(position):
                            result[0, :, t:t+crop_size, l:l+crop_size] += pred_crop[num, :, :, :].cpu()
                            voting_mask[:, :, t:t+crop_size, l:l+crop_size] += 1

                        # initialize the position queue
                        position = []

        # post processing
        out = result/voting_mask
        out = torch.clamp(out,min=0, max=1)
        out = out * 255
        gt = gt * 255

        # evaluation
        if eval_score is not None:
            score.update(eval_score(out, gt, 255), inp.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if save_vis == True:
            save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch))
            out = out.data.numpy()
            save_output_images(out, str(epoch), name, save_dir, epoch)

    if logger is not None:
        logger.info(' * Score is {s.avg:.3f}'.format(s=score))
    return score.avg
