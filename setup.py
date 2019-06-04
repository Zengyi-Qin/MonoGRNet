import os
# build Cython module
os.system('cd ./include/utils/ && make')

# build KITTI evaluation module
os.system('cd ./submodules/KittiEvaluation/ && make')

# download the pretrained model
if not os.path.exists('./data/model_2D.pkl'):
    os.system('wget https://cloud.tsinghua.edu.cn/f/d958b1da1c49496fa951/?dl=1 -O pretrained.tar.gz')
    os.system('tar -xf pretrained.tar.gz')
    os.system('rm pretrained.tar.gz')

print('Pretrained model downloaded.')

# check for Kitti data
if not os.path.exists('./data/KittiBox/training'):
    print('Kitti data not found. Please place it in ./data/KittiBox/training')
    exit()

print('Setup finished!')
