"""
Wrapper for CIFAR-10 dataset and TF model.

Written by Phil Ferriere

Loosely based on https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Note: we use the exact same format and folders as in the PyTorch sample

Licensed under the MIT License
"""

from __future__ import absolute_import, division, print_function
import os, sys, tarfile, pickle
import tensorflow as tf
import numpy as np
from urllib.request import urlretrieve

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000
NUM_TEST_SAMPLES = 10000
CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class CIFAR10():
    """TF data handler for CIFAR-10 dataset and model."""

    def __init__(self, batch_size=8, data_dir=None):
        """CIFAR-10 dataset and TF model constructor.
        Args:
            batch_size: dataset batch size.
        """
        self._train_data, self._train_labels = None, None
        self._test_data, self._test_labels = None, None
        self._batch_size = batch_size
        self.img_size = IMAGE_SIZE
        self.num_channels = NUM_CHANNELS
        self.num_classes = NUM_CLASSES
        self.train_len = NUM_TRAIN_SAMPLES
        self.test_len = NUM_TEST_SAMPLES
        self.data_dir = data_dir or "./test_data"
        self.cifar10_dir = os.path.join(self.data_dir, 'cifar-10-batches-py')
        self.cifar10_tarball = os.path.join(self.data_dir, CIFAR10_URL.split('/')[-1])
        self.maybe_download_and_extract()

    @property
    def train_data(self):
        if self._train_data is None:
            self._load('train')

        return self._train_data

    @property
    def train_labels(self):
        if self._train_labels is None:
            self._load('train')

        return self._train_labels

    @property
    def test_data(self):
        if self._test_data is None:
            self._load('test')

        return self._test_data

    @property
    def test_labels(self):
        if self._test_labels is None:
            self._load('test')

        return self._test_labels

    def _load(self, dataset='train'):
        """Load the data in memory.
        Args:
            dataset: string in ['train', 'test']
        """
        data, labels = None, None
        if dataset is 'train':
            files = [os.path.join(self.cifar10_dir, 'data_batch_%d' % i) for i in range(1, 6)]
        else:
            files = [os.path.join(self.cifar10_dir, 'test_batch')]

        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError('Failed to find file: ' + file)

        # Load the data from the batch files
        for file in files:
            with open(file, 'rb') as f:
                cifar10 = pickle.load(f, encoding='latin1')

            if labels is None:
                labels = np.array(cifar10['labels'])
            else:
                labels = np.concatenate((labels, cifar10['labels']), axis=0)

            if data is None:
                data = cifar10['data']
            else:
                data = np.concatenate((data, cifar10['data']), axis=0)

        # Adapt the format of the data to our convnet
        data = np.array(data, dtype=float) / 255.0
        data = data.reshape([-1, self.num_channels, self.img_size, self.img_size])
        data = data.transpose([0, 2, 3, 1])

        # One-hot encode labels (see https://stackoverflow.com/a/42874726)
        labels = np.eye(self.num_classes)[np.array(labels).reshape(-1)]

        if dataset is 'train':
            self._train_data, self._train_labels = data, labels
        else:
            self._test_data, self._test_labels = data, labels

    def model(self, inputs, mode='train'):
        """Build a simple convnet (BN before ReLU).
        Args:
            inputs: a tensor of size [batch_size, height, width, channels]
            mode: string in ['train', 'test']
        Returns:
            the last op containing the predictions
        Note:
            Best score
            Step:  7015 - Epoch: 18/20 - best batch acc: 0.8984 - loss: 1.5656
            Worst score
            Step:  7523 - Epoch: 20/20 - best batch acc: 0.7734 - loss: 1.6874
        """
        # Extract features
        training = (mode == 'train')
        with tf.variable_scope('conv1') as scope:
            conv = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='SAME')
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            bn = tf.nn.relu(bn)
            conv = tf.layers.conv2d(inputs=bn, filters=16, kernel_size=[3, 3], padding='SAME')
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            bn = tf.nn.relu(bn)
            pool = tf.layers.max_pooling2d(bn, pool_size=[2, 2], strides=2, padding='SAME', name=scope.name)

        with tf.variable_scope('conv2') as scope:
            conv = tf.layers.conv2d(inputs=pool, filters=32, kernel_size=[3, 3], padding='SAME')
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            bn = tf.nn.relu(bn)
            conv = tf.layers.conv2d(inputs=bn, filters=32, kernel_size=[3, 3], padding='SAME')
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            bn = tf.nn.relu(bn)
            pool = tf.layers.max_pooling2d(bn, pool_size=[2, 2], strides=2, padding='SAME', name=scope.name)

        with tf.variable_scope('conv3') as scope:
            conv = tf.layers.conv2d(inputs=pool, filters=32, kernel_size=[3, 3], padding='SAME')
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            bn = tf.nn.relu(bn)
            conv = tf.layers.conv2d(inputs=bn, filters=32, kernel_size=[3, 3], padding='SAME')
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            bn = tf.nn.relu(bn)
            pool = tf.layers.max_pooling2d(bn, pool_size=[2, 2], strides=2, padding='SAME', name=scope.name)

        # Classify
        with tf.variable_scope('fc') as scope:
            flat = tf.layers.flatten(pool)
            fc = tf.layers.dense(inputs=flat, units=32, activation=tf.nn.relu)
            softmax = tf.layers.dense(inputs=fc, units=self.num_classes, activation=tf.nn.softmax)

        return softmax

    def model2(self, inputs, mode='train'):
        """Build a simple convnet (ReLU before BN).
        Args:
            inputs: a tensor of size [batch_size, height, width, channels]
            mode: string in ['train', 'test']
        Returns:
            the last op containing the predictions
        Note:
            Best score
            Step:  7411 - Epoch: 20/20 - best batch acc: 0.8438 - loss: 1.6347
            Worst score
            Step:  7751 - Epoch: 20/20 - best batch acc: 0.8047 - loss: 1.6616
        """
        # Extract features
        training = (mode == 'train')
        with tf.variable_scope('conv1') as scope:
            conv = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            conv = tf.layers.conv2d(inputs=bn, filters=16, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            pool = tf.layers.max_pooling2d(bn, pool_size=[2, 2], strides=2, padding='SAME', name=scope.name)

        with tf.variable_scope('conv2') as scope:
            conv = tf.layers.conv2d(inputs=pool, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            conv = tf.layers.conv2d(inputs=bn, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            pool = tf.layers.max_pooling2d(bn, pool_size=[2, 2], strides=2, padding='SAME', name=scope.name)

        with tf.variable_scope('conv3') as scope:
            conv = tf.layers.conv2d(inputs=pool, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            conv = tf.layers.conv2d(inputs=bn, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            bn = tf.layers.batch_normalization(inputs=conv, training=training)
            pool = tf.layers.max_pooling2d(bn, pool_size=[2, 2], strides=2, padding='SAME', name=scope.name)

        # Classify
        with tf.variable_scope('fc') as scope:
            flat = tf.layers.flatten(pool)
            fc = tf.layers.dense(inputs=flat, units=32, activation=tf.nn.relu)
            softmax = tf.layers.dense(inputs=fc, units=self.num_classes, activation=tf.nn.softmax)

        return softmax

    def maybe_download_and_extract(self):
        """Download and extract the tarball from Alex Krizhevsky's website."""
        if not os.path.exists(self.cifar10_dir):

            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            def _progress(count, block_size, total_size):
                status_msg = '\r>> Downloading {} {:>3}%   '
                sys.stdout.write(status_msg.format(self.cifar10_tarball, float(count * block_size) / total_size * 100.0))
                sys.stdout.flush()

            file_path, _ = urlretrieve(CIFAR10_URL, self.cifar10_tarball, _progress)

            stat_info = os.stat(file_path)
            print('\nSuccessfully downloaded', file_path, stat_info.st_size, 'bytes.\n')

            tarfile.open(file_path, 'r:gz').extractall(self.data_dir)

