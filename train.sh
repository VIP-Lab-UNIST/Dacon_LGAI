CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 24 \
                                --epochs 100 \
                                --lr 1e-5 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.1 \
                                --resume ../Others/DW-GAN-Dehazing/weights/dehaze.pkl \
                                --save-dir ./logs/train/DWGAN/Pretrained/
                                # --save-dir ./logs/train/tmp

                                # --resume ./logs/train/DWGAN/202107161604/checkpoint_070.tar \
                                # 32
                                # --lr 1e-4 \
