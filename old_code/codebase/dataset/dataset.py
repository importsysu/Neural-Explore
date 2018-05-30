import math

import itertools

from dataset.read_cifar import Cifar10Reader, Cifar100Reader
from dataset.read_mnist import MnistReader

data_readers = {
    'cifar-10': Cifar10Reader,
    'cifar-100': Cifar100Reader,
    'mnist': MnistReader,
    'None': None
}

class Dataset(object):
    """Dataset class capsulates sufficient interfaces for trainer and tester.
       Dataset classes are initialized by a DataSetReader according to the name.
    """
    def __init__(self, dataset_name: str, role: str):
        """Initialize Dataset with dataset name of a dataset reader.

        Note:
            If the dataset reader is not in the configuration, the init function will raise an exception.

        Args:
            dataset_name: the name of this dataset.
        """
        if dataset_name not in data_readers:
            raise Exception('Dataset {} not defined'.format(dataset_name))
        elif dataset_name == 'None':
            pass
        else:
            self.name = dataset_name + "." + role
            self.reader = data_readers[dataset_name](role)
            self.meta_info = self.reader.get_meta_info()
            self.data = self.reader.get_all_data()
            self.size = self.reader.get_data_size()

    def get_batches(self, batch_size, num_batches=None, num_epoches = 1, shuffle=False, sample_type = 'cycle'):
        '''Get batches from the dataset

        Args:
            batch_size: batch size
            num_batches: the number of batches to generate in this function, 'None' for single epoch
            sample_type: This arg is not provided now
            shuffle: whether to shuffle dataset for each epoche
        '''
        num_batches_per_epoch = int(math.ceil(self.size / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch * num_epoches
        num_epoches = int(math.ceil(num_batches / num_batches_per_epoch))

        if sample_type == 'cycle':
            idx = itertools.cycle(range(self.size))
            for _ in range(num_batches):
                batch_idx = [next(idx) for k in range(batch_size)]
                batch_data = {
                    key: val[batch_idx]
                    for key, val in self.data.items()
                }
                yield batch_data
        elif sample_type == 'once':
            idx = itertools.chain(range(self.size))
            for _ in range(num_batches):
                batch_idx = [next(idx, None) for k in range(batch_size)]
                batch_idx = list(itertools.takewhile(lambda x: x is not None, batch_idx))
                batch_data = {
                    key: list(map(val.__getitem__, batch_idx))
                    for key, val in self.data.items()
                }
                yield batch_data
        else:
            raise ValueError('Sample-Type invalid')

    def split_validation(self, validation_size):
        assert validation_size <= self.size
        vali_set = Dataset('None', '')
        vali_set.meta_info = self.meta_info
        vali_set.data = {}
        vali_set.size = validation_size
        self.size -= validation_size
        for key, val in self.data.items():
            vali_set.data[key] = val[:validation_size]
            self.data[key] = val[validation_size:]
        return vali_set
