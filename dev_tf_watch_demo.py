"""
dev_tf_watch_demo.py

WeightWatcher demo (TensorFlow) to visualize training progress

Written by Phil Ferriere

Licensed under the MIT License

Note:
    This file is here for dev purposes only (to run the core notebook code through a debugger, if necessary) 
"""
from __future__ import absolute_import, division, print_function
import sys
import tensorflow as tf

from ww import builder_tf, watcher
from tf_cifar10 import CIFAR10

#
# Visualize Graph
#

# CIFAR10 dataset and model
batch_size = 128
cifar10 = CIFAR10(batch_size=batch_size)

# Inspect data and labels
watcher.show("cifar10.train_data", cifar10.train_data)
watcher.show("cifar10.train_labels", cifar10.train_labels)
watcher.show("cifar10.test_data", cifar10.test_data)
watcher.show("cifar10.test_labels", cifar10.test_labels)

# Setup TF "graphing" session
sess = tf.Session()

# Setup placeholders/vars
inputs = tf.placeholder(tf.float32, shape=(batch_size, cifar10.img_size, cifar10.img_size, cifar10.num_channels))

# Build model
predictions = cifar10.model(inputs)

# Run the initializer
sess.run(tf.global_variables_initializer())

# Convert TF graph to directed graph
dg = builder_tf.build_tf_graph(tf.get_default_graph(), sess, predictions.op.name) # Nodes (78)

# Dump graph
dg.list_layers()

# Simplify graph (DEBUGGING ONLY)
dg.simplify_graph(verbose=True)

# Dump graph
print("Simplified graph:")
dg.list_layers()

# Draw full graph
dg.draw_graph(simplify=False, verbose=True)

# Terminate "graphing" session
sess.close()
tf.reset_default_graph()

#
# Visualize Training Progress
#

# Setup TF training session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

# Setup placeholders/vars
inputs = tf.placeholder(tf.float32, shape=(batch_size, cifar10.img_size, cifar10.img_size, cifar10.num_channels))
outputs = tf.placeholder(tf.float32, shape=[batch_size, cifar10.num_classes])
g_step = tf.Variable(initial_value=0, trainable=False)

# Build model
predictions = cifar10.model(inputs)

# Setup loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=outputs))
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss, global_step=g_step)

# Setup metric
accurate_preds = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(outputs, axis=1))
accuracy = tf.reduce_mean(tf.cast(accurate_preds, tf.float32))

# Instantiate watcher
w = watcher.Watcher()

# Visual customizations
w.legend={"loss": "Training Loss", "accuracy": "Training Accuracy"}

# Run the initializer
sess.run(tf.global_variables_initializer())

# Run training loop on GPU
epochs = 20
_best_accuracy = -1
with tf.device('/gpu:0'): # Set to '/cpu:0' if you don't have a GPU
    for epoch in range(epochs):

        batches, _ = divmod(cifar10.train_len, batch_size)
        for batch in range(batches):

            # Fetch training samples
            _input = cifar10.train_data[batch*batch_size : (batch+1)*batch_size]
            _output = cifar10.train_labels[batch*batch_size : (batch+1)*batch_size]

            # Train model
            train_ops = [g_step, optimizer, loss, accuracy]
            step, _, _loss, _accuracy = sess.run(train_ops, feed_dict={inputs : _input, outputs : _output})

            # Print stats
            if batch & batch % 100 == 0:
                # BUBUG - Generates many `No handles with labels found to put in legend.` matplotlib  messages
                # w.step(step, loss=_loss, accuracy=_accuracy)
                # with w:
                #     w.plot(["loss"])
                #     w.plot(["accuracy"])
                if _accuracy > _best_accuracy:
                    _best_accuracy = _accuracy
                    status_msg = '\r>>Step: {:>5} - Epoch: {:>2}/{:>2} - best batch acc: {:.4f} - loss: {:.4f}'
                    sys.stdout.write(status_msg.format(step, epoch+1, epochs, _accuracy, _loss))
                    sys.stdout.flush()

# Terminate training session
sess.close()
