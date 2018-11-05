import os
import sys
import shutil
import unittest
import torch
import torchvision.models
import hiddenlayer as hl
from hiddenlayer import transforms as ht

# Create output directory in project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class TestPytorchGraph(unittest.TestCase):
    def test_graph(self):
        model = torchvision.models.vgg16()
        g = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
        g.save(os.path.join(OUTPUT_DIR, "pytorch_vgg16.pdf"))

        model = torchvision.models.resnet50()
        g = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
        g.save(os.path.join(OUTPUT_DIR, "pytorch_resnet50.pdf"))

        # Clean up
        shutil.rmtree(OUTPUT_DIR)


    def test_resnet_blocks(self):
        # Resnet101
        model = torchvision.models.resnet101()

        transforms = [
            # Fold Conv, BN, RELU layers into one
            ht.Fold("Conv > BatchNormalization > Relu", "ConvBnRelu"),
            # Fold Conv, BN layers together
            ht.Fold("Conv > BatchNormalization", "ConvBn"),
            # Fold bottleneck blocks
            ht.Fold("""
                ((ConvBnRelu > ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
                """, "BottleneckBlock", "Bottleneck Block"),
            # Fold residual blocks
            ht.Fold("""ConvBnRelu > ConvBnRelu > ConvBn > Add > Relu""",
                            "ResBlock", "Residual Block"),
            # Fold repeated blocks
            ht.FoldDuplicates(),
        ]

        # Display graph using the transforms above
        g = hl.build_graph(model, torch.zeros([1, 3, 224, 224]), transforms=transforms)
        g.save(os.path.join(OUTPUT_DIR, "pytorch_resnet_bloks.pdf"))

        # Clean up
        shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
