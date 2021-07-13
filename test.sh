CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                --data-dir ../Datasets \
                                --save-dir ./logs/tmp/gan05_mask/ \
                                --resume ./logs/train/gan05/202107122154/checkpoint_033.tar

                                # --save-dir ./logs/train/202107012237/stride_128 \
                                # --save-dir ./logs/test/202107012237/checkpoint_025/stride_32 \
