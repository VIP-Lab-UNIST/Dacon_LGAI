CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --data-dir ../Datasets \
                                --save-dir ./logs/train \
                                --crop-size 256 \
                                --batch-size 24 \
                                --epochs 500 \
                                --lr 1e-4