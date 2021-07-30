CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 704 1024 \
                                --save-interval 2 \
                                --batch-size 2 \
                                --epochs 100 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.005 \
                                --save-dir ./logs/train/khko_DWGAN_multiscale_multidis_/size704_1024/
                                # --save-dir ./logs/train/tmp

                                # --resume ./logs/train/DWGAN/202107161604/checkpoint_070.tar \
                                # 32
