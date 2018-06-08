"""
layer.py

Layer class.
Represents a framework-agnostic neural network layer in a directed graph.

Written by Waleed Abdulla, additions by Phil Ferriere

Licensed under the MIT License
"""
from __future__ import absolute_import, division, print_function

import numpy as np

class Layer():
    """Represents a framework-agnostic neural network layer in a directed graph."""

    def __init__(self, uid, name, op, output_shape=None, params=None):
        """
        uid: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation name.
        """
        self.id = uid # uid: 'VGG/Sequential[features]/Conv2d[0]/outputs/85'
        self.name = name # name: 'Conv'
        self.op = op # name: 'conv'
        self.repeat = 1
        if output_shape and type(output_shape) is not str:
            output_shape = str(output_shape)
        self.output_shape = output_shape
        self.params = params if params else {} # params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
        self._caption = ""

    @property
    def title(self):
        # Default
        title = self.name # title: 'Conv'

        if "kernel_shape" in self.params:
            # Kernel
            kernel = self.params["kernel_shape"]  # kernel: [3, 3]
            title += "x".join(map(str, kernel)) # title: 'Conv3x3'
        #         # Transposed
        #         if node.transposed:
        #             name = "Transposed" + name
        return title

    @property
    def caption(self):
        if self._caption:
            return self._caption

        caption = ""

        # Stride
        if "stride" in self.params:
            stride = self.params["stride"]
            if np.unique(stride).size == 1:
                stride = stride[0]
            if stride != 1:
                caption += "/{}".format(str(stride))
        return caption

    def __repr__(self):
        args = (self.op, self.name, self.id, self.title, self.repeat)
        f = "<Layer: op: {:15}, name: {:15}, id: {:50}, title: {:15}, repeat: {:2}"
        if self.output_shape:
            args += (self.output_shape,)
            f += ", shape: {:10}"
        if self.params:
            args += (str(self.params),)
            f += ", params: {:50}"
        f += ">"
        return f.format(*args)

    def __eq__(self, a):
        # TODO: not used right now
        assert isinstance(a, Layer)
        return hash(self.params) == hash(a.params)
