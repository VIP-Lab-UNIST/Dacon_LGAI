CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 16 \
                                --epochs 100 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.005 \
                                --save-dir ./logs/train/DWGAN_multiscale/

                                # --save-dir ./tmp
                                # --resume ./logs/train/DWGAN/DeformConv/KA_attn01/202107141819/checkpoint_026.tar \
                                # --save-dir ./logs/train/DWGAN/DeformConv/KA_attn34
                                # 32
