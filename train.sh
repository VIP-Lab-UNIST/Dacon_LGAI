CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 512 \
                                --batch-size 4 \
                                --epochs 100 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --gan_weight 0.5 \
                                --save-dir ./logs/train/mask_only_dis

                                # --save-dir ./logs/train/mask_only_output
                                # --save-dir ./logs/train/mask_gan00
                                # 32
