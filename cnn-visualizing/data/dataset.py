import random

from read_cifar import Cifar10Reader, Cifar100Reader

class Dataset(object):
	"""Dataset class capsulates sufficient interfaces for trainer and tester.
	   Dataset classes are initialized by a DataSetReader according to the name.
	"""

	data_readers={
		'cifar-10': Cifar10Reader,
		'cifar-100': Cifar100Reader
	}

	def __init__(self, dataset_name, role):
		"""Initialize Dataset with dataset name of a data reader.

		Note:
			If the data reader is not in the configuration, the init function will raise an exception.

		Args:
			dataset_name: the name of this dataset.
		"""
		self.name = dataset_name + "." + role
		if dataset_name not in Dataset.data_readers:
			raise Exception('Dataset {} not defined'.format(dataset_name))
		self.reader = Dataset.data_readers[dataset_name](role)
		self.meta_info = self.reader.get_meta_info()
		self.data = self.reader.get_all_data()
		self.size = self.reader.get_data_size()

	def get_batches(self, batch_size, num_batches=1, sample_type='default', shuffle=True):
		'''Get batches from the dataset

		Args:
			batch_size: batch size
			num_batches: the number of batches to generate in this function
			sample_type: only 'default' is valid currently
			shuffle: whether to shuffle data for each epoche
		'''
		num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))
		idxs = itertools.chain.from_iterable(
			random.shuffle(range(self.size)) if shuffle else range(self.size) for _ in range(num_epochs))
		for _ in range(num_batches):
			batch_idxs = tuple(itertools.islice(idxs, batch_size))
			batch_data = {}
			for key, val in self.data.items():
				batch_data[key] = list(map(val.__getitem__, batch_idxs))
			yield batch_data

#Unit Test
if __name__ == '__main__':
	cifar10_train = Dataset('cifar-10', 'train')
	cifar10_test = Dataset('cifar-10', 'test')
	cifar100_train = Dataset('cifar-100', 'train')
	cifar100_test = Dataset('cifar-100', 'test')

