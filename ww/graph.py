"""
graph.py

DirectedGraph class.
Tracks nodes and edges of a directed graph and supports basic operations on them.

Written by Waleed Abdulla, additions by Phil Ferriere

Licensed under the MIT License

Refs:
    graphviz Graph API
    @ http://graphviz.readthedocs.io/en/stable/api.html#graph
"""
from __future__ import absolute_import, division, print_function

from random import getrandbits
from graphviz import Digraph
from .layer import Layer

OUTLINE_COLOR = "#4080A0"  # "#484848"
FILL_COLOR = "#F0F0F0"  # "#E8E8E8"
FONT_COLOR = "#204050"  # "#000000"
FONT_NAME = "Verdana"  # "Times"
FONT_SIZE = "10"  # "8"

class DirectedGraph():
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""

    def __init__(self, prune_rules=None, fold_rules=None, group_rules=None, meaningful_ids=False):
        self.nodes = {}
        self.edges = []
        self.prune_rules = prune_rules
        self.fold_rules = fold_rules
        self.group_rules = group_rules
        self.meaningful_ids = meaningful_ids

        # Default style
        self.theme = {
            "outline_color": OUTLINE_COLOR,
            "fill_color": FILL_COLOR,
            "font_color": FONT_COLOR,
            "font_name": FONT_NAME,
            "font_size": FONT_SIZE,
        }

    def id(self, node):
        """Returns a unique node identifier. If the node has an id
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)

    def add_node(self, node):
        id = self.id(node)
        # assert(id not in self.nodes)
        self.nodes[id] = node

    def add_edge(self, node1, node2, label=None):
        self.edges.append((self.id(node1), self.id(node2), label))

    def add_edge_by_id(self, vid1, vid2, label=None):
        self.edges.append((vid1, vid2, label))

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
            in_node = self.nodes[k]
            self.add_edge(in_node, node, in_node.output_shape)
        for k in self.outgoing(nodes[-1]):
            self.add_edge(node, self[k], node.output_shape)
        # Remove the old nodes
        for n in nodes:
            self.remove(n)

    def fold(self, node1, node2):
        """Fold node1 in node2. Edges incoming to node1 are connected to
        node2 and node1 is removed. No new node is created.
        """
        # Connect edges incoming to node1 to node2 instead
        for k in self.incoming(node1):
            node = self.nodes[k]
            self.add_edge(node, node2, node.output_shape)
        # Get rid of node1
        self.remove(node1)

    def find_node(self, op):
        """Find the first node with the given OP.
        op: For example, "weights" or "biases".
        """
        for layer in self.nodes.values():
            if layer.op == op:
                return layer

        # Scanned all graph and found nothing
        return None

    def find_sequence(self, ops):
        """Find a sequence of nodes with the given OPs.
        ops: A string of ops separated by /. For example, "conv/relu".
        """
        ops = ops.split("/")
        for layer in self.nodes.values():
            layers = []
            for op in ops:
                if op != '*' and op != layer.op:
                    break
                layers.append(layer)
                # Get following layer
                layer = self.filter_one(self[self.outgoing(layer)])
                if not layer:
                    if layers == []:
                        break
                    else:
                        return layers
            else:
                # Finished all ops. We have a match
                return layers
        # Scanned all graph and found nothing
        return None

    def sequence_id(self, sequence):
        """Make up an ID for a sequence (list) of nodes.
        Note: `getrandbits()` is very uninformative as a "readable" ID. Here, we build a name
        such that when the mouse hovers over the drawn node in Jupyter, one can figure out
        which original nodes make up the sequence. This is actually quite useful.
        """
        if self.meaningful_ids:
            return "><".join([node.id for node in sequence])
        else:
            return getrandbits(64)

    def prune_nodes(self, verbose=False):
        """Prune semantically uninteresting layers.
        """
        if self.prune_rules is None:
            return False
        for prune in self.prune_rules:
            layer = self.find_node(prune)
            if layer:
                if verbose:
                    print("Pruning layer {}".format(layer))
                self.remove(layer)
                return True

    def collapse_nodes(self, verbose=False):
        """Collapse select sequences of layers into a single layer.
        """
        if self.fold_rules is None:
            return False
        for seq, fold in self.fold_rules.items():
            sequence = self.find_sequence(seq)
            if sequence:
                node1, node2 = sequence[0], sequence[1]
                if verbose:
                    print("Folding node {} into node {}".format(node1, node2))
                self.fold(node1, node2)
                return True

    def group_nodes(self, verbose=False):
        """Group select sequences of layers into combo layers.
        """
        if self.group_rules is None:
            return False
        for group in self.group_rules:
            sequence = self.find_sequence(group)
            if sequence:
                combo = Layer(uid=self.sequence_id(sequence),
                              name="/".join([l.title for l in sequence]),
                              op=group,
                              output_shape=sequence[-1].output_shape)
                combo._caption = "/".join(filter(None, [l.caption for l in sequence]))
                if verbose:
                    seq = ""
                    for layer in sequence:
                        seq = seq + "/" + str(layer)
                    print("Replacing sequence [{}] with combo {}".format(seq, combo))
                self.replace(sequence, combo)
                return True

        # Group layers of the same type together
        for node in self.nodes.values():
            node2 = self.filter_one(self[self.outgoing(node)])
            if node2 and node2.op == node.op:
                combo = Layer(uid=self.sequence_id([node, node2]),
                              name=node.title,
                              op=node.op,
                              output_shape=node.output_shape)
                combo._caption = node.caption
                combo.repeat = node.repeat + node2.repeat
                if verbose:
                    print("Replacing sequence [{}/{}] with combo {}".format(node, node2, combo))
                self.replace([node, node2], combo)
                return True
        return False

    def simplify_graph(self, verbose=False):
        """Simplify the graph using a sequence of pruning, collapsing, and grouping rules.
        """
        # TODO: copy the graph rather than modifying passed version
        while self.prune_nodes(verbose):
            pass
        while self.collapse_nodes(verbose):
            pass
        while self.group_nodes(verbose):
            pass

    def draw_graph(self, simplify=True, output_shapes=True, verbose=False):
        """Simplify the graph using a sequence of pruning, collapsing, and grouping rules.
        simplify: If True, simplify the graph before drawing it; otherwise, draw the graph unchanged.
        output_shapes: If True, print output shapes along edges; otherwise, don't.
        verbose: If True, print out the details of the graph simplification
        Ref:
            dot.attr(kw=None, _attributes=None, **attrs) to add a general or graph/node/edge attribute statement.
            @ http://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph.edge
            dot.node(name, label=None, _attributes=None, **attrs) to create a node
            @ http://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph.node
            dot.edge(tail_name, head_name, label=None, _attributes=None, **attrs) to create an edge between two nodes.
            @ http://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph.edge
        """
        # Simplify the graph to group repeated layers together
        if simplify:
            self.simplify_graph(verbose)

        # Convert to graphviz Digraph
        dot = Digraph()
        dot.attr("node", shape="box", style="filled", fillcolor="#e8e8e8", fontsize="10", margin="0.11,0")
        dot.attr("edge", fontsize="10")
        for k, n in self.nodes.items():
            if n.repeat == 1:
                if n.title == '+':
                    dot.node(str(k), n.title + "\n" + n.caption, shape='circle')
                else:
                    dot.node(str(k), n.title + "\n" + n.caption)
            else:
                with dot.subgraph(name="cluster {}".format(n.id)) as s:
                    s.attr(label="x{}".format(n.repeat),
                           labelloc="br", labeljust="r", fontsize="10",
                           style="dashed")
                    s.node(str(k), n.title + "\n" + n.caption)
        for a, b, label in self.edges:
            if output_shapes is False:
                label = None
            if label:
                label = "x".join(map(str, label))
            dot.edge(str(a), str(b), label)
        return dot

    def draw_graph_html(self, simplify=True, output_shapes=True, verbose=False):
        """Simplify the graph using a sequence of pruning, collapsing, and grouping rules.
        simplify: If True, simplify the graph before drawing it; otherwise, draw the graph unchanged.
        output_shapes: If True, print output shapes along edges; otherwise, don't.
        verbose: If True, print out the details of the TF graph simplification
        Ref:
            dot.attr(kw=None, _attributes=None, **attrs) to add a general or graph/node/edge attribute statement.
            @ http://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph.edge
            dot.node(name, label=None, _attributes=None, **attrs) to create a node
            @ http://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph.node
            dot.edge(tail_name, head_name, label=None, _attributes=None, **attrs) to create an edge between two nodes.
            @ http://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph.edge
        """
        # Simplify the graph to group repeated layers together
        if simplify:
            self.simplify_graph(verbose)

        # Convert to graphviz Digraph
        dot = Digraph()
        dot.attr("graph", splines="ortho", nodesep="2", color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"], fontcolor=self.theme["font_color"], fontname=self.theme["font_name"])
        dot.attr("node", shape="box", style="filled", margin="0,0", fillcolor=self.theme["fill_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"], fontcolor=self.theme["font_color"], fontname=self.theme["font_name"])
        dot.attr("edge", style="doted", color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"], fontcolor=self.theme["font_color"], fontname=self.theme["font_name"])

        for k, n in self.nodes.items():
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
        for a, b, label in self.edges:
            if output_shapes is False:
                label = None
            if label:
                label = "x".join(map(str, label))
            dot.edge(str(a), str(b), label)
        return dot

    def list_layers(self):
        """List the layers in the graph (for debugging purposes).
        """
        for layer in self.nodes.values():
            print(layer)


###########################################################################
# Unit Tests
###########################################################################
if __name__ == "__main__":
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
