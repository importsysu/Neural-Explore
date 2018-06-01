import os
import numpy as np
import tensorflow as tf

DATA_URL='http://www.deep-furnace.cn/data/%s/numpy/'
DATASETS=['mnist','cifar-10','cifar-100','fashion-mnist','SVHN','STL-10']

def download_dataset(data_name, out_dir=None):
    import requests
    if data_name not in DATASETS:
        raise ValueError('name "{}" not found'.format(data_name))
    base_url = DATA_URL % data_name
    file_list = requests.get(base_url+'files.txt').content.decode('ascii').strip().split('\n')
    print(str(len(file_list)) + ' files to download')

    # creating directory
    if out_dir is None:
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
        out_dir = os.path.join('datasets', data_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # download files
    for file_name in file_list:
        if os.path.exists(os.path.join(out_dir, file_name)):
            print(file_name + ' already exists.')
            continue
        print('Download from ' + base_url+file_name)
        req = requests.get(base_url+file_name)
        with open(os.path.join(out_dir,file_name), 'wb') as f:
            f.write(req.content)
        print('Finished.')
    print('Downloading Complete.')

def download_all(out_dir=None):
    for name in DATASETS:
        download_dataset(name)

def _default_image_dataset(split_name, data_dir):
    if split_name not in ['train', 'test']:
        raise ValueError('split name {} not valid'.format(split_name))
    image_path = os.path.join(data_dir, split_name + '_images.npy')
    label_path = os.path.join(data_dir, split_name + '_labels.npy')
    return tf.train.slice_input_producer([np.load(image_path), np.load(label_path)])

def load_mnist(split_name, data_dir=None):
    if data_dir is None:
        data_dir = './datasets/mnist'
    return _default_image_dataset(split_name, data_dir)

def load_cifar10(split_name, data_dir=None):
    if data_dir is None:
        data_dir = './datasets/cifar-10'
    return _default_image_dataset(split_name, data_dir)

def load_cifar100(split_name, data_dir=None):
    if data_dir is None:
        data_dir = './datasets/cifar-100'
    image_path = os.path.join(data_dir, split_name + '_images.npy')
    fine_label_path = os.path.join(data_dir, split_name + '_fine_labels.npy')
    coarse_label_path = os.path.join(data_dir, split_name + '_coarse_labels.npy')
    return tf.train.slice_input_producer([np.load(image_path), np.load(fine_label_path), np.load(coarse_label_path)])

def load_fashion_mnist(split_name, data_dir=None):
    if data_dir is None:
        data_dir = './datasets/fashion-mnist'
    return _default_image_dataset(split_name, data_dir)

def load_stl10(split_name, data_dir=None):
    if data_dir is None:
        data_dir = './datasets/STL-10'
    if split_name in ['train', 'test']:
        return _default_image_dataset(split_name, data_dir)
    elif split_name == 'unlabeled':
        image_path = os.path.join(data_dir, split_name + '_images.npy')
        image_tensor = tf.train.slice_input_producer([np.load(image_path)])
        return image_tensor
    else:
        raise ValueError('Invalid split name')

def load_svhn(split_name, data_dir=None):
    if data_dir is None:
        data_dir = './datasets/SVHN'
    return _default_image_dataset(split_name, data_dir)

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == 'download_all':
        download_all()
    if cmd == 'download':
        download_dataset(sys.argv[2])
