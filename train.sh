CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 32 \
                                --epochs 500 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.01 \
                                --gan_weight 0. \
                                --save-dir ./logs/tmp/
                                # --save-dir ./logs/train/multiscale/instance_norm

                                # --save-dir ./logs/train/basline/