CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 256 \
                                --batch-size 8 \
                                --epochs 500 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.01 \
                                --save-dir ./logs/tmp/
                                # --save-dir ./logs/train/baseline/deep_discriminator/
                                # --resume ../Others/MSBDN-DFF/models/model.pkl \

                                # --save-dir ./logs/train/basline/