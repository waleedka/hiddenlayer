"""
TF Graph importer

Written by Phil Ferriere

Licensed under the MIT License
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from .graph import Graph, Node
from . import transforms as ht


FRAMEWORK_TRANSFORMS = [
    ht.Prune("ConcatIgnore"),  # TODO: no need to create *Ignore nodes then deleting them
    ht.Prune("DropoutIgnore"),
    ht.Prune("FlattenIgnore"),
    ht.Prune("MeanIgnore"),
    ht.Prune("PadIgnore"),
    ht.Prune("Biases"),
    ht.Prune("Weights"),
    ht.Prune("Const"),
    ht.Prune("Variable"),
    ht.Prune("Assign"),
    ht.Prune("AssignSub"),
    ht.Rename(op=r"Conv2D", to=r"Conv"),
    ht.Rename(op=r"FusedBatchNorm", to=r"BatchNormalization"),
    ht.Rename(op=r"(\w+)V\d", to=r"\1"),
    ht.Fold("Conv > BiasAdd", "__first__"),
    ht.Fold("Linear > BiasAdd", "__first__"),
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


def import_graph(hl_graph, tf_graph, output=None, verbose=False):
    """Convert TF graph to directed graph
    tfgraph: A TF Graph object.
    output: Name of the output node (string).
    verbose: Set to True for debug print output
    """
    # Get clean(er) list of nodes
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    graph_def = tf.graph_util.remove_training_nodes(graph_def)

    # Dump list of TF nodes (DEBUG only)
    if verbose:
        dump_tf_graph(tf_graph, graph_def)

    # Loop through nodes and build the matching directed graph
    # TODO: Repace the transformations below with Transform objects
    for tf_node in graph_def.node:
        # Operation type and name
        if "weights" in tf_node.name.lower():
            op = "Weights"
        elif "biases" in tf_node.name.lower():
            op = "Biases"
        elif "dropout" in tf_node.name.lower():
            if "dropout/div" in tf_node.name.lower() or "dropout/mul" in tf_node.name.lower():
                op = "Dropout"
            else:
                op = "DropoutIgnore"
        elif "flatten" in tf_node.name.lower():
            if "strided_slice" in tf_node.name.lower() or "/shape" in tf_node.name.lower():
                op = "FlattenIgnore"
            else:
                op = "Flatten"
        elif "paddings" in tf_node.name.lower():
            op = "PadIgnore"
        elif "reduction_indices" in tf_node.name.lower():
            op = "MeanIgnore"
        elif "concat/axis" in tf_node.name.lower():
            op = "ConcatIgnore"
        elif "dense" in tf_node.name.lower() and "matmul" in tf_node.name.lower():
            op = "Linear"
        else:
            op = tf_node.op
        name = None
        uid = tf_node.name  # TODO

        # Inputs
        inputs = tf_node.input

        # Shape
        shape = tf.graph_util.tensor_shape_from_node_def_name(tf_graph, tf_node.name).as_list()

        # Parameters
        # At this stage, we really only care about two parameters:
        # 1/ the kernel size used by convolution layers, 2/ the stride used by pooling layers

        # 1/ The kernel size is actually not stored in the convolution tensor but in its weight input.
        # The weights input has the shape [shape=[kernel, kernel, in_channels, filters]]
        # So we must fish for it
        params = {} # None
        if op == "Conv2D":
            kernel_shape = tf.graph_util.tensor_shape_from_node_def_name(tf_graph, tf_node.input[1])
            kernel_shape = [int(a) for a in kernel_shape]
            params["kernel_shape"] = kernel_shape[0:2]
        elif op == "MaxPool" or op == "AvgPool":
            # 2/ the stride used by pooling layers
            # See https://stackoverflow.com/questions/44124942/how-to-access-values-in-protos-in-tensorflow
            if 'ksize' in tf_node.attr.keys():
                kernel_shape = [int(a) for a in tf_node.attr['ksize'].list.i]
                params["kernel_shape"] = kernel_shape[1:3]
            # if 'strides' in node.attr.keys():
            #     strides = [int(a) for a in node.attr['strides'].list.i]
            #     params["stride"] = strides[1:3]

        # Add layer
        layer = Node(uid=uid, name=name, op=op, output_shape=shape, params=params)
        hl_graph.add_node(layer)

        # Add edges
        for target_node in graph_def.node:
            target_inputs = target_node.input
            if uid in target_node.input:
                hl_graph.add_edge_by_id(uid, target_node.name, shape)
    return hl_graph
