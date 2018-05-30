import os

import tensorflow as tf
import urllib.request
import gzip
import numpy

from dataset.config import MnistConfig

mnist_constants = {
    'SOURCE_URL': 'http://yann.lecun.com/exdb/mnist/',
    'IMAGE_SIZE': 28,
    'PIXEL_DEPTH': 255,
    'NUM_CHANNELS': 1
}

class MnistReader(object):
    """Mnist Reader reads mnist dataset from downloaded files.
    """
    def __init__(self, role):
        self.__dict__.update(mnist_constants)

        if role not in ['train', 'test']:
            raise ValueError('Invalid Role: {}'.format(role))

        if role == 'train':
            train_data_filename = self.check_download('train-images-idx3-ubyte.gz')
            train_labels_filename = self.check_download('train-labels-idx1-ubyte.gz')
            self.data = self.extract_data(train_data_filename, 60000)
            self.labels = self.extract_labels(train_labels_filename, 60000)
        else:
            test_data_filename = self.check_download('t10k-images-idx3-ubyte.gz')
            test_labels_filename = self.check_download('t10k-labels-idx1-ubyte.gz')
            self.data = self.extract_data(test_data_filename, 10000)
            self.labels = self.extract_labels(test_labels_filename, 10000)

        self.datasize = self.data.shape[0]

    def check_download(self, filename):
        """Download the dataset from Yann's website, unless it's already here."""
        WORK_DIRECTORY = MnistConfig.DATA_PATH
        if not tf.gfile.Exists(WORK_DIRECTORY):
            tf.gfile.MakeDirs(WORK_DIRECTORY)
        filepath = os.path.join(WORK_DIRECTORY, filename)
        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.SOURCE_URL + filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.Size()
            tf.logging.info('Successfully downloaded {}, {} bytes'.format(filename, size))

        return filepath

    def extract_data(self, filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        tf.logging.info('Extracting {}'.format(filename))
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(self.IMAGE_SIZE * self.IMAGE_SIZE * num_images)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = (data - (self.PIXEL_DEPTH / 2.0)) / self.PIXEL_DEPTH
            data = data.reshape(num_images, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        return data

    def extract_labels(self, filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        return labels

    def get_meta_info(self):
        return {}

    def get_all_data(self):
        return {'data': self.data, 'label': self.labels}

    def get_data_size(self):
        return self.datasize

