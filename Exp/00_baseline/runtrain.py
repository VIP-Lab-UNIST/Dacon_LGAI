import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.system("python3 main_init.py train -d ../../data -s 256 --batch-size 24\
        --epochs 500 --lr 1e-4")

#os.system("python3 main.py train -d ../training_set -s 1024 --batch-size 32\
#        --epochs 250 --lr 0.0001 --momentum 0.9 --step 50")
#        --resume ./runs/train/202003191922/checkpoint_040.pth.tar\

