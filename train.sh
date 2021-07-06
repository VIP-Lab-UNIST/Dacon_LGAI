CUDA_VISIBLE_DEVICES=0,1 python3 main.py train \
                                --data-dir Datasets \
                                --save-dir ./logs/train \
                                --crop-size 512 \
                                --batch-size 8 \
                                --epochs 100 \
                                --lr 1e-4