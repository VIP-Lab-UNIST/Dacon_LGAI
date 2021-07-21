CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 20 \
                                --epochs 150 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.05 \
                                --resume ./logs/train/DWGAN/202107151742/checkpoint_061.tar \
                                --save-dir ./logs/train/khko_DWGAN_multiscale_multidis/

                                # --save-dir ./logs/tmp
                                # --gan_weight 0.005 \
                                # --save-dir ./logs/train/DWGAN/DeformConv/KA_attn34
                                # 32
