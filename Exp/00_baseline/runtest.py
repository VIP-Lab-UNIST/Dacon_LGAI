import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.system("python3 main_init.py test -d ../../data -s 256 --batch-size 24")
