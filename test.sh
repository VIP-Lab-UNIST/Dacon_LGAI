CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                --data-dir ../Datasets \
                                --save-dir ./logs \
                                --resume .outputs/checkpoint_best.tar
