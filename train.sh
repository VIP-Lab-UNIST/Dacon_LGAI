CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 1024 \
                                --batch-size 1 \
                                --epochs 100 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --save-dir ./logs/tmp 

                                # --save-dir ./logs/train \
