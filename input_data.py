"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import, division, print_function

import gzip
import os
import glob
from sklearn.model_selection import train_test_split

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

INPUT_SHAPE = (1500, 1500, 1)
SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"

class DataSet(object):
    def __init__(self, images, labels, fake_data=False, skip_reshape=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], f"images.shape: {images.shape} labels.shape: {labels.shape}"
            self._num_examples = images.shape[0]
            if not skip_reshape:
                # Convert shape from [num examples, rows, columns, depth]
                # to [num examples, rows*columns] (assuming depth == 1)
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(data_dir, fake_data=False, one_hot=False, num_classes=10):
    class DataSets(object):
        pass

    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    data = []
    labels = []
    all_files = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
    #print(all_files)

    for file in all_files:
        # One-hot encoding for labels
        label = [0, 0, 0]
        if os.path.dirname(file).endswith('GoogleDrive'):
            label[0] = 1
        elif os.path.dirname(file).endswith('GoogleDoc'):
            label[1] = 1
        elif os.path.dirname(file).endswith('Youtube'):
            label[2] = 1
        labels.append(label)

        arr = numpy.load(file) #[:,:,dimensions_to_use] #eliminated dimensions to use
        data.append(arr.reshape(INPUT_SHAPE)) 

    labels = numpy.asarray(labels)
    data = numpy.asarray(data).reshape(-1,*INPUT_SHAPE)

    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    data_sets.train = DataSet(X_train, y_train)
    data_sets.validation = DataSet(X_val, y_val)
    data_sets.test = DataSet(X_test, y_test)
    return data_sets