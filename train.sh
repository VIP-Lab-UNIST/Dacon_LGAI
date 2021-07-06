CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 512 \
                                --batch-size 8 \
                                --epochs 100 \
                                --lr 1e-4 \
                                --data-dir ./Datasets \
                                --save-dir ./logs/train 
