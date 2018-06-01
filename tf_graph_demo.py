"""
tf_graph_demo.py

Render TF graphs

Written by Phil Ferriere

Licensed under the MIT License

Refs:
    TensorFlow-Slim image classification model library
    @ https://github.com/tensorflow/models/tree/master/research/slim
    Copyright 2016 The TensorFlow Authors. All Rights Reserved.

    TF-Slim Walkthrough
    @ https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb
    Copyright 2016 The TensorFlow Authors. All Rights Reserved.

    TF Slim Backbones
    @ https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
    @ https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg_test.py
    @ https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py
    @ https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1_test.py
    Copyright 2016 The TensorFlow Authors. All Rights Reserved.

    Tensor class code
    @ E:\toolkits.win\anaconda3-5.1.0\envs\dlwin36\Lib\site-packages\tensorflow\python\framework\ops.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ww import builder_tf

#
# VGG 16
#
from tensorflow.contrib.slim.nets import vgg

# Build TF graph and convert it
with tf.Session() as sess:
    # Setup input placeholder
    batch_size = 1
    height, width = vgg.vgg_16.default_image_size, vgg.vgg_16.default_image_size
    inputs = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    # Build model
    predictions, model = vgg.vgg_16(inputs)

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Convert TF graph to directed graph
    dg = builder_tf.build_tf_graph(tf.get_default_graph(), sess, predictions.op.name) # Nodes (110)

# Dump graph
print("Original graph:")
dg.list_layers()

# Original graph:
# <Layer: op: data, name: Placeholder, id: Placeholder, title: Placeholder>
# <Layer: op: weights, name: Weights, id: vgg_16/conv1/conv1_1/weights, title: Weights>
# <Layer: op: biases, name: Biases, id: vgg_16/conv1/conv1_1/biases, title: Biases>
# <Layer: op: conv, name: Conv2D, id: vgg_16/conv1/conv1_1/Conv2D, title: Conv2D>
# <Layer: op: biasadd, name: BiasAdd, id: vgg_16/conv1/conv1_1/BiasAdd, title: BiasAdd>
# <Layer: op: relu, name: Relu, id: vgg_16/conv1/conv1_1/Relu, title: Relu>
# <Layer: op: weights, name: Weights, id: vgg_16/conv1/conv1_2/weights, title: Weights>
# <Layer: op: biases, name: Biases, id: vgg_16/conv1/conv1_2/biases, title: Biases>
# <Layer: op: conv, name: Conv2D, id: vgg_16/conv1/conv1_2/Conv2D, title: Conv2D>
# <Layer: op: biasadd, name: BiasAdd, id: vgg_16/conv1/conv1_2/BiasAdd, title: BiasAdd>
# <Layer: op: relu, name: Relu, id: vgg_16/conv1/conv1_2/Relu, title: Relu>
# <Layer: op: maxpool, name: MaxPool, id: vgg_16/pool1/MaxPool, title: MaxPool>

# Draw full graph
# dg.draw_graph(simplify=False)

# Draw simplified graph
# dg.draw_graph(simplify=True)

# Simplify graph (DEBUGGING ONLY)
dg.simplify_graph(verbose=True)

# Dump graph
print("Simplified graph:")
dg.list_layers()

# Draw full graph
dg.draw_graph(simplify=False, verbose=True)

#
# ResNet
#
from tensorflow.contrib.slim.nets import resnet_v1

# Build TF graph and convert it
with tf.Session() as sess:
    # Setup input placeholder
    batch_size = 1
    height, width = resnet_v1.resnet_v1.default_image_size, resnet_v1.resnet_v1.default_image_size
    inputs = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    # Build model
    predictions, end_points = resnet_v1.resnet_v1_50(inputs)

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Convert TF graph to directed graph
    dg = builder_tf.build_tf_graph(tf.get_default_graph(), sess, predictions.op.name) # Nodes (292)

# Dump graph
dg.list_layers()

# Simplify graph (DEBUGGING ONLY)
dg.simplify_graph(verbose=True)

# Dump graph
print("Simplified graph:")
dg.list_layers()

# Draw full graph
dg.draw_graph(simplify=False, verbose=True)


