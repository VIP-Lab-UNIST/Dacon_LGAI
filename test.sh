for epoch in {25..20..1}
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py test \
                                        --data-dir ../Datasets \
                                        --save-dir ./logs/train/khko_HINet/loss/PSNRLoss/CosineAnnealingWarmRestarts/202107261923/outputs \
                                        --resume ./logs/train/khko_HINet/loss/PSNRLoss/CosineAnnealingWarmRestarts/202107261923/checkpoint_0$epoch.tar

    done

# CUDA_VISIBLE_DEVICES=0 python3 main.py test \
#                                 --data-dir ../Datasets \
#                                 --save-dir ./logs/tmp \
#                                 --resume ./logs/train/202107012237/checkpoint_025_tensor.tar

#                                 # --save-dir ./logs/train/202107012237/stride_128 \
#                                 # --save-dir ./logs/test/202107012237/checkpoint_025/stride_32 \
