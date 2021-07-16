CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                --data-dir ../Datasets \
                                --save-dir ./logs/train/DWGAN/202107151742/test \
                                --resume ./logs/train/DWGAN/202107151742/checkpoint_059.tar

                                # --save-dir ./logs/train/202107012237/stride_128 \
                                # --save-dir ./logs/test/202107012237/checkpoint_025/stride_32 \
