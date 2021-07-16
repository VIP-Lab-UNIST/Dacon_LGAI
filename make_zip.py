import os
import zipfile
from glob import glob
from os.path import basename
import warnings
warnings.filterwarnings("ignore")

root = './logs/train/DWGAN/202107151742/test'
# root = './logs/train/DWGAN/DeformConv/Resnet/202107141821/test/'
# root = './logs/train/DWGAN/DeformConv/Resnet_KA_attn01/202107141818/test/'

files = glob(root+'/**/*.png', recursive=True)
zip_file = os.path.join(root, 'output.zip')

fantasy_zip = zipfile.ZipFile(zip_file, 'w')
for img in files:
    print(basename(img))
    fantasy_zip.write(img, basename(img).replace('_input', ''))
fantasy_zip.close()