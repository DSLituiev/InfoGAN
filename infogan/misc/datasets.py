import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np


class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

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

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class MnistDataset(object):
    def __init__(self, proportions = None):
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        if proportions is None:
            self.train = dataset.train
        else:
            assert len(proportions) == 10
            sup_images = []
            sup_labels = []
            rnd_state = np.random.get_state()
            np.random.seed(0)
            len_train = len(dataset.train.labels)
            from collections import Counter
            label_counts = Counter( dataset.train.labels )
            coef = min( (label_counts[nn] // proportions[nn] for nn in range(10) ) )
            for cat in range(10):
                num_cat = coef * proportions[cat]
                print( "digit:", cat, "frequency:", num_cat)
                ids = np.where(dataset.train.labels == cat)[0]
                np.random.shuffle(ids)
                sup_images.extend(dataset.train.images[ids[:num_cat]])
                sup_labels.extend(dataset.train.labels[ids[:num_cat]])
            np.random.set_state(rnd_state)
            self.train = Dataset(
                np.asarray(sup_images),
                np.asarray(sup_labels),
            )

        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data
