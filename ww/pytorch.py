import random
import numpy as np
import torch
from graphviz import Digraph

from . import common

# Requires PyTorch 0.4+
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
    "Gemm": "Linear",
    "BatchNormalization": "BatchNorm",
}


def pytorch_id(node):
    """Returns a unique ID for a node."""
    # After ONNX simplification, the scopeName is not unique anymore 
    # so append node outputs to gurantee uniqueness
    return node.scopeName() + "/outputs/" + "/".join([o.uniqueName() for o in node.outputs()])


def build_pytorch_graph(model, args, input_names=None):
    # TODO: add input names to graph

    # Initialize an empty directed graph to store the layers
    dg = common.DirectedGraph()

    # Run the Pytorch graph to get a trace and generate a graph from it
    trace, out = torch.jit.get_trace_graph(model, args)
    torch.onnx._optimize_trace(trace, False)
    graph = trace.graph()

    # Loop through nodes and build graph layers
    for node in graph.nodes():
        # Op
        op = node.kind().replace("onnx::", "")
        name = op
        
        # Map to internal name
        op = PYTORCH_OP_MAP.get(op, op)
        name = PYTORCH_NAME_MAP.get(name, name)
        
        # Parameters
        params = {k: node[k] for k in node.attributeNames()}
        
        # Inputs/outputs
        inputs = [i.unique() for i in node.inputs()]
        outputs = [o.unique() for o in node.outputs()]
        
        # Add layer
        layer = common.Layer(uid=pytorch_id(node),
                      name=name, op=op, params=params)
        dg.add_node(layer)
        
        # Add edges
        for target_node in graph.nodes():
            target_inputs = [i.unique() for i in target_node.inputs()]
            if set(outputs) & set(target_inputs):
                dg.add_edge_by_id(pytorch_id(node), pytorch_id(target_node))
    return dg


