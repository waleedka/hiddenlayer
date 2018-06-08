"""
builder_tf.py

TF Graph builder

Written by Phil Ferriere

Licensed under the MIT License
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from .graph import DirectedGraph
from .layer import Layer

# Mapping framework specific names to internal names
TF_OP_MAP = {
    "Add": "add",
    "ArgMax": "argmax",
    "AvgPool": "avgpool",
    "Biases": "ignore", # "ignore", "biases",
    "BiasAdd": "biasadd",
    "ConcatV2": "concat",
    "ConcatIgnore": "ignore", # "ignore", "concatignore",
    "Const": "ignore",
    "Conv2D": "conv",
    "Dropout": "dropout",
    "DropoutIgnore": "ignore", # "ignore", "dropoutignore",
    "FC": "fc",
    "Flatten": "flatten",
    "FlattenIgnore": "ignore", # "ignore", "flattenignore",
    "FusedBatchNorm": "bn",
    "MaxPool": "maxpool",
    "Mean": "mean",
    "MeanIgnore": "ignore", # "ignore", "meanignore"
    "Pad": "pad",
    "PadIgnore": "ignore", # "ignore", "padignore"
    "Placeholder": "data",
    "Relu": "relu",
    "Softmax": "softmax",
    "Squeeze": "squeeze",
    "Weights": "ignore", # "ignore", "weights"
}

TF_NAME_MAP = {
    "Add": "+",
    "FusedBatchNorm": "BatchNorm",
    "Conv2D": "Conv",
    "ConcatV2": "Concat",
    "Placeholder": "Data",
}

# Nodes to prune
# TF_PRUNE_RULES = [
#   "weights",
#   "biases",
#   "concatignore",
#   "dropoutignore",
#   "meanignore",
#   "padignore"
# ]
TF_PRUNE_RULES = ["ignore"]

# Sequences of nodes to collapse together
TF_FOLD_RULES = {
    # "biasadd/relu" : "relu",
    # "biasadd/squeeze" : "squeeze",
    # "biasadd/add" : "add",
    "biasadd/*": "*"
}

# Sequences of layers to group together
TF_GROUP_RULES = [
    "fc/relu/dropout",
    "fc/relu",
    "fc/softmax",
    "conv/bn/relu",
    "conv/relu/bn",
    "conv/relu",
    "conv/bn",
]

def dump_tf_graph(tfgraph, tfgraphdef):
    """List all the nodes in a TF graph.
    tfgraph: A TF Graph object.
    tfgraphdef: A TF GraphDef object.
    """
    print("Nodes ({})".format(len(tfgraphdef.node)))
    f = "{:15} {:59} {:20} {}"
    print(f.format("kind", "scopeName", "shape", "inputs"))
    for node in tfgraphdef.node:
        scopename = node.name
        kind = node.op
        inputs = node.input
        shape = tf.graph_util.tensor_shape_from_node_def_name(tfgraph, scopename)
        print(f.format(kind, scopename, str(shape), inputs))


def build_tf_graph(tfgraph, sess, output, verbose=False):
    """Convert TF graph to directed graph
    tfgraph: A TF Graph object.
    sess: A TF Session object.
    output: Name of the output node (string).
    verbose: Set to True for debug print output
    """
    # Get clean(er) list of nodes
    tfgraphdef = tfgraph.as_graph_def(add_shapes=True)
    tfgraphdef = tf.graph_util.remove_training_nodes(tfgraphdef)
    tfgraphdef = tf.graph_util.convert_variables_to_constants(sess, tfgraphdef, [output])

    # Dump list of TF nodes (DEBUG only)
    if verbose:
        dump_tf_graph(tfgraph, tfgraphdef)

    # Initialize an empty directed graph to store the layers
    dg = DirectedGraph(TF_PRUNE_RULES, TF_FOLD_RULES, TF_GROUP_RULES, meaningful_ids=True)

    # Loop through nodes and build the matching directed graph
    for node in tfgraphdef.node:
        # print(node)
        # Operation type and name
        if "weights" in node.name.lower():
            op = "Weights"
        elif "biases" in node.name.lower():
            op = "Biases"
        elif "dropout" in node.name.lower():
            if "dropout/div" in node.name.lower() or "dropout/mul" in node.name.lower():
                op = "Dropout"
            else:
                op = "DropoutIgnore"
        elif "flatten" in node.name.lower():
            if "strided_slice" in node.name.lower() or "/shape" in node.name.lower():
                op = "FlattenIgnore"
            else:
                op = "Flatten"
        elif "paddings" in node.name.lower():
            op = "PadIgnore"
        elif "reduction_indices" in node.name.lower():
            op = "MeanIgnore"
        elif "concat/axis" in node.name.lower():
            op = "ConcatIgnore"
        elif "dense" in node.name.lower() and "matmul" in node.name.lower():
            op = "FC"
        else:
            op = node.op
        name = op
        op = TF_OP_MAP.get(op, op)
        name = TF_NAME_MAP.get(name, name)
        uid = node.name  # PyTorch name: 'Conv'

        # Inputs
        inputs = node.input

        # Shape
        shape = tf.graph_util.tensor_shape_from_node_def_name(tfgraph, node.name)

        # Parameters
        # At this stage, we really only care about two parameters:
        # 1/ the kernel size used by convolution layers, 2/ the stride used by pooling layers

        # 1/ The kernel size is actually not stored in the convolution tensor but in its weight input.
        # The weights input has the shape [shape=[kernel, kernel, in_channels, filters]]
        # So we must fish for it
        params = {} # None
        if op == "conv":
            kernel_shape = tf.graph_util.tensor_shape_from_node_def_name(tfgraph, node.input[1])
            kernel_shape = [int(a) for a in kernel_shape]
            params["kernel_shape"] = kernel_shape[0:2]
        elif op == "maxpool" or op == "avgpool":
            # 2/ the stride used by pooling layers
            # See https://stackoverflow.com/questions/44124942/how-to-access-values-in-protos-in-tensorflow
            if 'ksize' in node.attr.keys():
                kernel_shape = [int(a) for a in node.attr['ksize'].list.i]
                params["kernel_shape"] = kernel_shape[1:3]
            # if 'strides' in node.attr.keys():
            #     strides = [int(a) for a in node.attr['strides'].list.i]
            #     params["stride"] = strides[1:3]

        # Add layer
        layer = Layer(uid=uid, name=name, op=op, output_shape=shape, params=params)
        dg.add_node(layer)

        # Add edges
        for target_node in tfgraphdef.node:
            target_inputs = target_node.input
            if uid in target_node.input:
                dg.add_edge_by_id(uid, target_node.name, shape)

    return dg
