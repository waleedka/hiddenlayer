"""
builder_pytorch.py

PyTorch Graph builder

Written by Waleed Abdulla

Licensed under the MIT License
"""
from __future__ import absolute_import, division, print_function

from .graph import DirectedGraph
from .layer import Layer

# Requires PyTorch 0.4
import torch
from distutils.version import LooseVersion
assert LooseVersion(torch.__version__) >= LooseVersion("0.4")

# Mapping framework specific names to internal names
PYTORCH_OP_MAP = {
    "Conv": "conv",
    "BatchNormalization": "bn",
    "Gemm": "linear",
    "Relu": "relu",
    "MaxPool": "maxpool",
    "Dropout": "dropout",
}

PYTORCH_NAME_MAP = {
    "Add": "+",
    "Gemm": "Linear",
    "BatchNormalization": "BatchNorm",
}

# Nodes to prune
# PYTORCH_PRUNE_RULES = []

# Sequences of nodes to collapse together
# PYTORCH_FOLD_RULES = {}

# Sequences of layers to group together
PYTORCH_GROUP_RULES = [
    "linear/relu/dropout",
    "linear/relu",
    "conv/bn/relu",
    "conv/relu",
    "conv/bn",
    # ("Conv2D+weights+biases", "conv"),
    # ("Assign", ""),
]

# BUGBUG
#
# Adding the two lines above:
#
# ("Conv2D+weights+biases", "conv"),
# ("Assign", ""),
#
#  to PYTORCH_GROUP_RULES yields the following:
#
# E:\repos\litegraph-dev\ww\graph.py in simplify_graph(self, verbose)
#     228         while self.collapse_nodes(verbose):
#     229             pass
# --> 230         while self.group_nodes(verbose):
#     231             pass
#     232
#
# E:\repos\litegraph-dev\ww\graph.py in group_nodes(self, verbose)
#     189             return False
#     190         for group in self.group_rules:
# --> 191             sequence = self.find_sequence(group)
#     192             if sequence:
#     193                 combo = Layer(uid=self.sequence_id(sequence),
#
# E:\repos\litegraph-dev\ww\graph.py in find_sequence(self, ops)
#     125         ops: A string of ops separated by /. For example, "conv/relu".
#     126         """
# --> 127         ops = ops.split("/")
#     128         for layer in self.nodes.values():
#     129             layers = []
#
# AttributeError: 'tuple' object has no attribute 'split'
#
# Waleed, is the intent to use them as folding rules?

def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))

def pytorch_id(node):
    """Returns a unique ID for a node."""
    # After ONNX simplification, the scopeName is not unique anymore
    # so append node outputs to gurantee uniqueness
    return node.scopeName() + "/outputs/" + "/".join([o.uniqueName() for o in node.outputs()])

def build_pytorch_graph(model, args, input_names=None, verbose=False):
    # TODO: add input names to graph

    # Initialize an empty directed graph to store the layers
    dg = DirectedGraph(group_rules=PYTORCH_GROUP_RULES)

    # Run the Pytorch graph to get a trace and generate a graph from it
    trace, out = torch.jit.get_trace_graph(model, args)
    torch.onnx._optimize_trace(trace, False)
    graph = trace.graph()

    # Dump list of nodes (DEBUG only)
    if verbose:
        dump_pytorch_graph(graph)

    # Loop through nodes and build graph layers
    for node in graph.nodes():  # node: %85 : Float(1, 64, 224, 224) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%0, %1, %2), scope: VGG/Sequential[features]/Conv2d[0]
        # Op
        op = node.kind().replace("onnx::", "")  # op: 'Conv'
        name = op  # name: 'Conv'

        # Map to internal name
        op = PYTORCH_OP_MAP.get(op, op)  # op: 'conv'
        name = PYTORCH_NAME_MAP.get(name, name)  # name: 'Conv'

        # Parameters
        params = {k: node[k] for k in node.attributeNames()}  # params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}

        # Inputs/outputs
        inputs = [i.unique() for i in node.inputs()]  # inputs: <class 'list'>: [0, 1, 2]
        outputs = [o.unique() for o in node.outputs()]  # outputs: <class 'list'>: [85]

        # Add layer
        layer = Layer(uid=pytorch_id(node), name=name, op=op, params=params)
        dg.add_node(layer)

        # Add edges
        for target_node in graph.nodes():
            target_inputs = [i.unique() for i in target_node.inputs()]  # target_inputs: <class 'list'>: [0, 1, 2]
            if set(outputs) & set(target_inputs):  # outputs: <class 'list'>: [85], target_inputs: <class 'list'>: [85, 3, 4, 5, 6]
                dg.add_edge_by_id(pytorch_id(node), pytorch_id(target_node))  # pytorch_id(node): 'VGG/Sequential[features]/Conv2d[0]/outputs/85', pytorch_id(target_node): 'VGG/Sequential[features]/BatchNorm2d[1]/outputs/86/87/88/batch_norm_dead_output-89/batch_norm_dead_output-90'
    return dg
