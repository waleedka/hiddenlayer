import os
import sys
import shutil
import unittest
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import hiddenlayer as hl

# Hide GPUs. Not needed for this test.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

OUTPUT_DIR = "test_output"

class TestTensorFlow(unittest.TestCase):
    def test_graph(self):
        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.vgg.vgg_16(tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.Graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_vgg16", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.resnet_v1.resnet_v1_50(
                    tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.Graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_resnet50", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.inception.inception_v1(
                    tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.Graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_inception", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.alexnet.alexnet_v2(
                    tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.Graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_alexnet", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.overfeat.overfeat(
                    tf.placeholder(tf.float32, shape=(1, 231, 231, 3)))
                dot = hl.Graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_overfeat", directory=OUTPUT_DIR, cleanup=True)

        import tf_cifar10
        with tf.Session():
            with tf.Graph().as_default() as g:
                tf_cifar10.CIFAR10().model(inputs=tf.placeholder(tf.float32, shape=(8, 32, 32, 3)))
                dot = hl.Graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_cifar10", directory=OUTPUT_DIR, cleanup=True)

        # Clean up
        # TODO: shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
