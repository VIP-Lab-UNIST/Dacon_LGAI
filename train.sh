CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 24 \
                                --epochs 500 \
                                --lr 2e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --regular_weight 0.1 \
                                --save-dir ./logs/train/AECRNet/lr/CosineAnnealingLR
                                # --save-dir ./logs/tmp/

                                # --save-dir ./logs/train/basline/