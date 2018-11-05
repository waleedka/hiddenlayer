import os
import sys
import shutil
import unittest
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib import layers
import hiddenlayer as hl

# Hide GPUs. Not needed for this test.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Create output directory in project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "test_data")

def lrelu(x):
    return tf.maximum(0.01 * x, x)

class TrafficSignsModel():
    """Model taken from my traffic signs recognition repo.
    https://github.com/waleedka/traffic-signs-tensorflow
    """
    def conv(self, input, num_outputs, name=None):
        return layers.convolution2d(
            input, num_outputs=num_outputs, kernel_size=(5, 5), stride=(1, 1), 
            padding="SAME", activation_fn=lrelu,
            normalizer_fn=layers.batch_norm
        )
    
    def pool(self, input):
        return layers.max_pool2d(input, kernel_size=(2, 2), 
                                 stride=(2, 2), padding="SAME")
        
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Global step counter
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # Placeholders
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], name="images")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            # Layers
            self.conv1 = self.conv(self.images, 8)
            self.pool1 = self.pool(self.conv1)
            self.conv2 = self.conv(self.pool1, 12)
            self.pool2 = self.pool(self.conv2)
            self.conv3 = self.conv(self.pool2, 16)
            self.pool3 = self.pool(self.conv3)
            self.flat = layers.flatten(self.pool3)
# TODO             self.h1 = layers.fully_connected(self.flat, 200, lrelu)
            self.logits = layers.fully_connected(self.flat, 62, lrelu)
            # Convert one-hot vector to label index (int). 
            self.predicted_labels = tf.argmax(self.logits, 1)
            # Loss
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels, name="test_name"))
            # Training Ops
            self.train = tf.train.AdamOptimizer(learning_rate=0.001)\
                                 .minimize(self.loss, global_step=self.global_step)
            self.init = tf.global_variables_initializer()
            # Create session
            self.session = tf.Session()
            # Run initialization op
            self.session.run(self.init)


class TestTensorFlow(unittest.TestCase):
    def test_graph(self):
        m = TrafficSignsModel()

        dot = hl.build_graph(m.graph).build_dot()
        dot.format = 'pdf'
        dot.render("tf_traffic_signs", directory=OUTPUT_DIR, cleanup=True)

        # Import CIFAR from the demos folder
        sys.path.append("../demos")
        import tf_cifar10

        with tf.Session():
            with tf.Graph().as_default() as g:
                tf_cifar10.CIFAR10(data_dir=DATA_DIR).model(inputs=tf.placeholder(tf.float32, shape=(8, 32, 32, 3)))
                dot = hl.build_graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_cifar10", directory=OUTPUT_DIR, cleanup=True)


class TestSlimModels(unittest.TestCase):
    def test_graph(self):
        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.vgg.vgg_16(tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.build_graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_vgg16", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.resnet_v1.resnet_v1_50(
                    tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.build_graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_resnet50", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.inception.inception_v1(
                    tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.build_graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_inception", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.alexnet.alexnet_v2(
                    tf.placeholder(tf.float32, shape=(1, 224, 224, 3)))
                dot = hl.build_graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_alexnet", directory=OUTPUT_DIR, cleanup=True)

        with tf.Session():
            with tf.Graph().as_default() as g:
                nets.overfeat.overfeat(
                    tf.placeholder(tf.float32, shape=(1, 231, 231, 3)))
                dot = hl.build_graph(g).build_dot()
                dot.format = 'pdf'
                dot.render("tf_overfeat", directory=OUTPUT_DIR, cleanup=True)

        # Clean up
        shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
