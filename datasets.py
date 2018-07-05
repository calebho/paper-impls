"""Download and load various datasets"""

import gzip
import io
import mnist
import numpy as np
import os
import pickle
import requests
import tarfile
import torch

from torch.utils.data import Dataset

CIFAR10_DIR = 'datasets/cifar-10-batches-py/'
MNIST_DIR = 'datasets/mnist/'
MNIST_URLS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]


class MyDataset(Dataset):
    def __init__(self, x, y=None):
        if y is not None:
            assert x.shape[0] == y.shape[0], 'Data and label shape mismatch'
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])
        else:
            return torch.from_numpy(self.x[idx])


def get_cifar10():
    if not os.path.exists(CIFAR10_DIR):
        # Download the data
        resp = requests.get(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        with io.BytesIO(resp.content) as f:
            tar_file = tarfile.open(mode='r:gz', fileobj=f)
            tar_file.extractall('datasets/')

    # Load the training data
    train_data = []
    train_labels = []
    for batch_num in range(1, 6):
        fname = CIFAR10_DIR + 'data_batch_{}'.format(batch_num)
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            train_data.append(data[b'data'])
            train_labels.append(data[b'labels'])
    train_data = np.concatenate(train_data).astype(np.float32)
    train_labels = np.concatenate(train_labels).astype(np.float32)

    # Load the test data
    with open(CIFAR10_DIR + 'test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        test_data = data[b'data'].astype(np.float32)
        test_labels = np.array(data[b'labels'], dtype=np.float32)
    return (MyDataset(train_data, train_labels),
            MyDataset(test_data, test_labels))


def get_mnist():
    if not os.path.exists(MNIST_DIR):
        os.mkdir(MNIST_DIR, 0o700)
        for url in MNIST_URLS:
            resp = requests.get(url)
            gz_fname = os.path.basename(url)
            fname, _ = os.path.splitext(gz_fname)
            with io.BytesIO(resp.content) as inp:
                with gzip.open(inp, 'rb') as data, open(MNIST_DIR + fname, 'wb') as out:  # yapf: disable
                    out.write(data.read())

    loader = mnist.MNIST(MNIST_DIR, return_type='numpy')
    train_data, train_labels = loader.load_training()
    train_data = train_data.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    test_data, test_labels = loader.load_testing()
    test_data = test_data.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    get_cifar10()
    get_mnist()
