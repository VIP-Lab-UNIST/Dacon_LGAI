for epoch in {25..20..1}
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                        --data-dir ../Datasets \
                                        --save-dir ./logs/train/DWGAN_multiscale/202107191153/outputs \
                                        --resume ./logs/train/DWGAN_multiscale/202107191153/checkpoint_0$epoch.tar

                                        # --save-dir ./logs/train/202107012237/stride_128 \
                                        # --save-dir ./logs/test/202107012237/checkpoint_025/stride_32 \
    done