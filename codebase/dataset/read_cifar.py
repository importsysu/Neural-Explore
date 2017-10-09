import os
import pickle
import numpy

from dataset.config import CifarConfig

def unpickle(file):
    """Unpickle files (Python3 style)."""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict


class Cifar100Reader(object):
    """Class Cifar-100 Reader reads dataset from pickled cifar-100 dataset
    
    Args:
        role: the role of this dataset part, including train, test, validation etc. Invalid role will raise exception.
    """
    def __init__(self, role):
        self.role=role
        if role == 'train':
            rawdata = unpickle(os.path.join(CifarConfig.DATA_PATH, 'cifar-100-python/train'))
        elif role == 'test':
            rawdata = unpickle(os.path.join(CifarConfig.DATA_PATH, 'cifar-100-python/test'))
        else:
            raise ValueException('Role is invalid for Cifar-100: {}'.format(role))
        self.data_size = len(rawdata[b'fine_labels'])
        self.data = {
            'coarse_label': rawdata[b'coarse_labels'],
            'fine_label': rawdata[b'fine_labels'],
            'data': rawdata[b'data']
        }

    def get_all_data(self):
        return self.data

    def get_data_size(self):
        return self.data_size

    def get_meta_info(self):
        return unpickle(os.path.join(CifarConfig.DATA_PATH, 'cifar-100-python/meta'))


class Cifar10Reader(object):
    """Class Cifar-100 Reader reads dataset from pickled cifar-100 dataset
    
    Args:
        role: the role of this dataset part, including train, test, validation etc. Invalid role will raise exception.
    """
    def __init__(self, role):
        self.role=role
        if role == 'train':
            temp = [unpickle(os.path.join(CifarConfig.DATA_PATH, 'cifar-10-batches-py/data_batch_%d' % (i + 1))) for i in range(5)]
            keys = {b'labels', b'data'}
            rawdata = {key: numpy.concatenate(tuple(temp[i][key] for i in range(5)), axis=0) for key in keys}
        elif role == 'test':
            rawdata = unpickle(os.path.join(CifarConfig.DATA_PATH, 'cifar-10-batches-py/test_batch'))
        else:
            raise ValueException('Role is invalid for Cifar-100: {}'.format(role))
        self.data_size = len(rawdata[b'labels'])
        self.data = {
            'labels': rawdata[b'labels'],
            'data': rawdata[b'data']
        }

    def get_all_data(self):
        return self.data

    def get_data_size(self):
        return self.data_size

    def get_meta_info(self):
        return unpickle(os.path.join(CifarConfig.DATA_PATH, 'cifar-100-python/meta'))
