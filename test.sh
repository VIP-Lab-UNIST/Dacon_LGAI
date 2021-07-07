CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                --data-dir ../Datasets \
                                --save-dir ./logs/tmp \
                                --resume ./logs/train/202107012237/checkpoint_025_tensor.tar

                                # --save-dir ./logs/train/202107012237/stride_128 \
                                # --save-dir ./logs/test/202107012237/checkpoint_025/stride_32 \
