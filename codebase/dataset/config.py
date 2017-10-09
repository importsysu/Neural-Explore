import os

class BasicConfig(object):
    DATA_DIR='/Users/zuo/Documents/ResearchMaterials/'

class MnistConfig(object):
    DATA_PATH=os.path.join(BasicConfig.DATA_DIR, 'mnist')
    IMAGE_SIZE=28
    PIXEL_DEPTH=3

class CifarConfig(object):
    DATA_PATH=os.path.join(BasicConfig.DATA_DIR, 'cifar')
