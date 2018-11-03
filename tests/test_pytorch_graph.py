import os
import sys
import shutil
import unittest
import torch
import torchvision.models
import hiddenlayer as hl

OUTPUT_DIR = "test_output"


class TestPytorchGraph(unittest.TestCase):
    def test_graph(self):
        model = torchvision.models.vgg16()
        layout = hl.Graph(model, torch.zeros([1, 3, 224, 224]))
        layout.save(os.path.join(OUTPUT_DIR, "pytorch_vgg16.pdf"))

        model = torchvision.models.resnet50()
        layout = hl.Graph(model, torch.zeros([1, 3, 224, 224]))
        layout.save(os.path.join(OUTPUT_DIR, "pytorch_resnet50.pdf"))

        # Clean up
        # TODO: shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
