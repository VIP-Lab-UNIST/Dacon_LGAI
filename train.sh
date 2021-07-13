CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 512 \
                                --batch-size 8 \
                                --epochs 100 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --gan_weight 0.0 \
                                --save-dir ./logs/tmp

                                # --save-dir ./logs/train/mask_gan00
                                # 32
