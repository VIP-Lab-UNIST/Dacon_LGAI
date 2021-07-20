
for epoch in {100..125..5}
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                        --data-dir ../Datasets \
                                        --save-dir ./logs/train/multiscale/202107181534/outputs \
                                        --resume ./logs/train/multiscale/202107181534/checkpoint_$epoch.tar

                                        # --save-dir ./logs/train/202107012237/stride_128 \
                                        # --save-dir ./logs/test/202107012237/checkpoint_025/stride_32 \
    done
