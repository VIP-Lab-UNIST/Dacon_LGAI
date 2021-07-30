CUDA_VISIBLE_DEVICES=0 python3 main.py train \
                                --crop-size 896 1280 \
                                --batch-size 1 \
                                --epochs 10000 \
                                --lr 1e-4 \
                                --data-dir ../Datasets \
                                --ssim_weight 0.2 \
                                --perc_weight 0.001 \
                                --gan_weight 0.\
                                --save-dir ./logs/train/khko_MSBDN/size896_1280
                                # --save-dir ./logs/tmp/
                                # --resume ../Others/MSBDN-DFF/models/model.pkl \

                                # --save-dir ./logs/train/basline/