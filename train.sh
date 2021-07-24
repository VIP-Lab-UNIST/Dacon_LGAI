CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 8 \
                                --epochs 500 \
                                --lr 2e-4 \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --data-dir ../Datasets \
                                --save-dir ./logs/train/khko_HINet/loss/PSNRLoss/CosineAnnealingWarmRestarts
                                # --save-dir ./logs/tmp/

                                # --resume ./logs/others/HINet/HINet-GoPro.pth \
                                # --save-dir ./logs/train/basline/