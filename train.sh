CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 1 \
                                --epochs 500 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0. \
                                --perc_weight 0. \
                                --edge_weight 0.05 \
                                --save-dir ./logs/tmp/
                                # --save-dir ./logs/train/khko/

                                # --save-dir ./logs/train/basline/