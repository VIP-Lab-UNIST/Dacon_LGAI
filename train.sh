CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 704 1024 \
                                --save-interval 2 \
                                --batch-size 2 \
                                --epochs 100 \
                                --lr 5e-5 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.005 \
                                --resume ./logs/train/DWGAN/202107151742/checkpoint_059.tar \
                                --save-dir ./logs/train/tmp
                                # --save-dir ./logs/train/DWGAN/size704_1024/

                                # --lr 1e-4 \
                                # 32
