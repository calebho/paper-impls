"""Download and load various datasets"""

import gzip
import io
import mnist
import numpy as np
import os
import pickle
import requests
import tarfile

CIFAR10_DIR = 'datasets/cifar-10-batches-py/'
MNIST_DIR = 'datasets/mnist/'
MNIST_URLS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]


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

    # Load the test data
    with open(CIFAR10_DIR + 'test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        test_data = data[b'data']
        test_labels = data[b'labels']
    return (np.concatenate(train_data), np.concatenate(train_labels),
            test_data, np.array(test_labels))


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
    test_data, test_labels = loader.load_testing()
    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    c10_train_data, c10_train_labels, c10_test_data, c10_test_labels = \
        get_cifar10()
    print(c10_train_data)
    print(c10_train_data.shape)
    print(c10_train_labels)
    print(c10_train_labels.shape)
    print(c10_test_data)
    print(c10_test_data.shape)
    print(c10_test_labels)
    print(c10_test_labels.shape)

    print(get_mnist())
