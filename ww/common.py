import math
import random
import io
import time
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib

COLOR_OUTLINE = "#4080A0"
COLOR_FILL = "#F0F0F0"
COLOR_FONT = "#204050"

NODE_COMBOS = [
    "linear/relu/dropout",
    "linear/relu",
    "conv/bn/relu",
    "conv/relu",
    "conv/bn",
    ("Conv2D+weights+biases", "conv"),
    ("Assign", ""),
]



class DirectedGraph():
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        
    def id(self, node):
        """Returns a unique node identifier. If the node has an id 
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)
    
    def add_node(self, node):
        self.nodes[self.id(node)] = node
    
    def add_edge(self, node1, node2):
        self.edges.append((self.id(node1), self.id(node2)))

    def add_edge_by_id(self, vid1, vid2):
        self.edges.append((vid1, vid2))

    def outgoing(self, node):
        """Returns IDs of nodes connecting out of the given node."""
        return [e[1] for e in self.edges if e[0] == self.id(node)]
    
    def incoming(self, node):
        """Returns IDs of nodes connecting to the given node."""
        return [e[0] for e in self.edges if e[1] == self.id(node)]
    
    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.nodes.get(k) for k in key]
        else:
            return self.nodes.get(key)
    
    def filter_one(self, nodes):
        if len(nodes) == 1:
            return nodes[0]
        else:
            return None
    
    def remove(self, node):
        """Remove a node and its edges."""
        k = self.id(node)
        self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
        del self.nodes[k]
        
    def replace(self, nodes, node):
        """Replace nodes with node. Edges incoming to nodes[0] are connected to 
        the new node, and nodes outgoing from nodes[-1] become outgoing from 
        the new node."""
        nodes = list(nodes)
        # Add new node and edges
        self.add_node(node)
        for k in self.incoming(nodes[0]):
            self.add_edge(self[k], node)
        for k in self.outgoing(nodes[-1]):
            self.add_edge(node, self[k])
        # Remove the old nodes
        for n in nodes:
            self.remove(n)
        


class Layer():
    """Represents a framework-agnostic neural network layer in a directed graph."""
    
    def __init__(self, uid, name, op, params=None):
        """
        uid: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation name.
        """
        self.id = uid
        self.name = name
        self.op = op
        self.repeat = 1
        self.params = params or {}
        self._caption = ""

        
    @property
    def title(self):
        # Default
        title = self.name
        
        if "kernel_shape" in self.params:
            # Kernel
            kernel = self.params["kernel_shape"]
            title += "x".join(map(str, kernel))
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
        
    def __eq__(self, a):
        # TODO: not used right now
        assert isinstance(a, Layer)
        return hash(self.params) == hash(a.params)
    
def find_sequence(dg, ops):
    """Find a sequence of nodes with the given OPs.
    dg: DirectedGraph object
    ops: A string of ops separated by /. For example, "conv/relu".
    """
    ops = ops.split("/")
    for key, layer in dg.nodes.items():
        layers = []
        for op in ops:
            if layer.op != op:
                break
            layers.append(layer)
            # Get following layer
            layer = dg.filter_one(dg[dg.outgoing(layer)])
            if not layer:
                break
        else:
            # Finished all ops. We have a match
            return layers
    # Scanned all graph and found nothing
    return None
    

# Group nodes
def group_nodes(dg):
    # Sequences of layers to group together
    GROUPS = [
        "linear/relu/dropout",
        "linear/relu",
        "conv/bn/relu",
        "conv/relu",
        "conv/bn",
    ]
    
    # Group select sequences of layers into combo layers
    for group in GROUPS:
        sequence = find_sequence(dg, group)
        if sequence:
            combo = Layer(uid=random.getrandbits(64),
                          name="/".join([l.title for l in sequence]),
                          op=group)
            combo._caption = "/".join(filter(None, [l.caption for l in sequence]))
            dg.replace(sequence, combo)
            return True

    # Group layers of the same type together
    for key, node in dg.nodes.items():
        node2 = dg.filter_one(dg[dg.outgoing(node)])
        if node2 and node2.op == node.op:
            combo = Layer(uid=random.getrandbits(64),
                          name=node.title,
                          op=node.op)
            combo._caption = node.caption
            combo.repeat = node.repeat + node2.repeat
            dg.replace([node, node2], combo)
            return True
    return False


def simplify_graph(graph):
    # TODO: copy the graph rather than modifying passed version
    while group_nodes(graph):
        pass


def draw_graph(dg):
    # Simplify the graph to group repeated layers together
    simplify_graph(dg)

    # Convert to graphviz Digraph
    dot = Digraph()
    dot.attr("graph", splines="ortho", nodesep="2", color=COLOR_OUTLINE, fontcolor=COLOR_FONT,
             fontsize="10", fontname="Verdana")
    dot.attr("node", shape="box", style="filled", fillcolor=COLOR_FILL, color=COLOR_OUTLINE,
             fontsize="10", margin="0,0", fontcolor=COLOR_FONT, fontname="verdana")
    dot.attr("edge", style="doted", color=COLOR_OUTLINE)
    
    for k, n in dg.nodes.items():
        label = "<tr><td cellpadding='10'><b>{}</b></td></tr>".format(n.title)
        if n.caption:
            label += "<tr><td>{}</td></tr>".format(n.caption)
        if n.repeat > 1:
            label += "<tr><td align='right' cellpadding='2'>x{}</td></tr>".format(n.repeat)
        label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
        
        if True or n.repeat == 1:  # TODO:
            if n.title == "Add":  # todo: match on OP
                dot.node(str(k), "<<b>+</b>>", fontsize="12", shape="circle")
            else:
                dot.node(str(k), label)
        # else:
        #     with dot.subgraph(name="cluster {}".format(n.id)) as s:
        #         s.attr(label="x{}".format(n.repeat),
        #                labelloc="br", labeljust="r",
        #                style="dashed")
        #         s.node(str(k), label)
    for a, b in dg.edges:
        dot.edge(str(a), str(b))
    return dot


if __name__ == "__main__":
    ###########################################################################
    # Unit Tests
    ###########################################################################
    import unittest

    class TestCommon(unittest.TestCase):
        def test_directed_graph(self):
            g = DirectedGraph()
            g.add_node("a")
            g.add_node("b")
            g.add_node("c")
            g.add_edge("a", "b")
            g.add_edge("b", "c")

            assert g[g.incoming("b")[0]] == "a"
            assert g[g.outgoing("b")[0]] == "c"
            g.replace(["b"], "x")
            assert sorted(list(g.nodes.values())) == sorted(["a", "c", "x"])
            assert g[g.incoming("x")[0]] == "a"
            assert g[g.outgoing("x")[0]] == "c"

    unittest.main()

