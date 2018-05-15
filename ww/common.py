import math
import random
import io
import itertools
import time
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import IPython.display
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

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


###############################################################################
# Misc logging functions
###############################################################################

def to_data(value):
    """Converts PyTorch tensors to numpy arrays or a scalar."""
    if value.__class__.__module__.startswith("torch"):
        import torch
        if isinstance(value, torch.nn.parameter.Parameter):
            value = value.data
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                value = value.detach()
            value = value.numpy().copy()
        # If 0-dim array, convert to scalar
        if not value.shape:
            value = value.item()
    return value


def show(*args):
    """Like print(), but recognizes tensors and arrays and show
    more details about them.

    Example:
        show("My Tensor", my_tensor)
    
    prints something like:
        My Tensor  float32 (10, 3, 224, 224)  min: 0.0  max: 1.0
    """
    s = ""
    for a in args:
        # Convert PyTorch tensors to Numpy arrays
        a = to_data(a)

        if isinstance(a, np.ndarray):
            # Numpy Array
            s += ("\t" if s else "") + "Tensor  {} {}  min: {:.3f}  max: {:.3f}".format(
                a.dtype, a.shape, a.min(), a.max())
            print(s)
            s = ""
        elif isinstance(a, list):
            s += ("\t" if s else "") + "list    len: {}  {}".format(len(a), a[:50])
        else:
            s += (" " if s else "") + str(a)
    if s:
        print(s)


def norm(image):
    return (image - image.min()) / (image.max() - image.min())


def show_images(images, titles=None, cols=5, **kwargs):
    """
    images: A list of images. I can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    # The images param can be a list or an array

    titles = titles or [""] * len(images)
    rows = math.ceil(len(images) / cols)
    height_ratio = 1.2 * (rows/cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(11, 11 * height_ratio))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [norm(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = norm(image)
        plt.title(title, fontsize=9)
        plt.imshow(image, cmap="Greys_r", **kwargs)
        i += 1
    plt.tight_layout(h_pad=0, w_pad=0)


###############################################################################
# Watcher Class
###############################################################################

class Watcher():
    """Tracks training progress and visualizes it.
    For example, use it to track the training and validation loss and accuracy
    and plot them.
    """
    
    DEFAULT_OPTIONS = {
        "fig_width": 12,  # inches
    }
    
    def __init__(self, plots=None):
        self.log = {}
        self.legend = {}
        self._in_context = False  # TODO: rename
        self.options = self.DEFAULT_OPTIONS

    def step(self, step, **kwargs):
        self.log[step] = {k:to_data(v) for k, v in kwargs.items()}

    def __enter__(self):
        self._in_context = True
        self.calls = []
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False
        # Clear output
        IPython.display.clear_output(wait=True)
        
        width = self.DEFAULT_OPTIONS['fig_width']
        fig = plt.figure(figsize=(width, width/3 * len(self.calls)))
        gs = matplotlib.gridspec.GridSpec(len(self.calls), 1)
    
        for i, c in enumerate(self.calls):
            getattr(self, c[0])(*c[1], **c[2], fig=fig, subplot_spec=gs[i])
        plt.show()
    
    def __getattribute__(self, name):
        if name in ["plot", "image_reel"] and self._in_context:
            def wrapper(*args, **kwargs):
                self.calls.append((name, args, kwargs))
            return wrapper
        else:
            return object.__getattribute__(self, name) 
    
    def get_ax(self, ax):
        # Axes
        if ax is None:
            w = self.options['fig_width']
            ax = plt.subplots(1, figsize=(w, w // 3))[1]
        return ax

    def plot(self, keys=None, title="", fig=None, subplot_spec=None):

        # Steps: extract from log
        steps = sorted(self.log.keys())
        
        # Keys: use provided value or scan the log and extract unique keys
        keys = keys or set(itertools.chain(*[list(s.keys()) for s in self.log.values()]))

        # Values: Parse log into a dict of key: [list of values with None for missing values]
        values = {}
        for k in keys:
            values[k] = [self.log[s].get(k) for s in steps]
        
        # Keep numerical values and filter out the rest
        keys = list(filter(lambda k: isinstance(values[k][0], (int, float)), keys))
        
        # Figure: use given figure or the pyplot current figure
        fig = fig or plt.gcf()

        # Divide area into a grid
        if subplot_spec is not None:
            ax = fig.add_subplot(subplot_spec)
        else:
            ax = fig.add_subplot(1, 1, 1)

        # Display
        ax.set_title("Step: {}".format(steps[-1]), fontsize=9)
        for k in keys:
            ax.plot(steps, values[k], label=self.legend.get(k))
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.set_xlabel("Steps")

    def image_reel(self, keys=None, title="", fig=None, subplot_spec=None):

        # Steps: extract from log
        steps = sorted(self.log.keys())
        
        # Keys: use provided value or scan the log and extract unique keys
        keys = keys or set(itertools.chain(*[list(s.keys()) for s in self.log.values()]))
        
        # Values: Parse log into a dict of key: [list of values with None for missing values]
        values = {}
        for k in keys:
            values[k] = [self.log[s].get(k) for s in steps]
        
        # Keep image values and filter out the rest
        keys = list(filter(lambda k: isinstance(values[k][0], np.ndarray), keys))
        
        # Figure: use given figure or the pyplot current figure
        fig = fig or plt.gcf()

        # How many images to show
        rows = len(keys)
        cols = 5

        # Divide area into a grid
        if subplot_spec is not None:
            gs = matplotlib.gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=subplot_spec)
        else:
            gs = matplotlib.gridspec.GridSpec(rows, cols)
        
        for i, key in enumerate(keys):
            for j, image in enumerate(values[key][-cols:]):
                ax = fig.add_subplot(gs[i, j])
                ax.axis('off')
#                 ax.set_title("step {}".format(step))
                ax.imshow(image)

    def hist(self, keys=None, title="", fig=None, subplot_spec=None):
        limit = 10  # max steps to show

        # Steps: extract from log
        steps = sorted(self.log.keys())
        
        # Values: Parse log into a dict of key: [list of values with None for missing values]
        key = keys[0]  # TODO: supporting one key for now
        values = {s:self.log[s].get(key) for s in steps}
        assert isinstance(list(values.values())[0], np.ndarray)
        
        # Figure: use given figure or the pyplot current figure
        fig = fig or plt.gcf()

        # How many images to show
        rows = 1
        cols = 1

        # Divide area into a grid
        if subplot_spec is not None:
            gs = matplotlib.gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=subplot_spec)
        else:
            gs = matplotlib.gridspec.GridSpec(rows, cols)
        
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax.view_init(30, -80)

        # Compute histograms
        verts = []
        colors = []
        for i, s in enumerate(steps[-limit:]):
            hist, edges = np.histogram(values[s])
            # X is bin centers
            x = np.diff(edges)/2 + edges[:-1]
            # Y is hist values
            y = hist
            x = np.concatenate([x[0:1], x, x[-1:]])
            y = np.concatenate([[0], y, [0]])
            # Z is step
            z = np.ones_like(y) * s

            # Ranges
            if i == 0:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
            x_min = np.minimum(x_min, x.min())
            x_max = np.maximum(x_max, x.max())
            y_min = np.minimum(y_min, y.min())
            y_max = np.maximum(y_max, y.max())

            ax.plot(x, z, y, color=[0, 0, .9, (i+1)/limit])
            # verts.append([(x[0], 0)] + list(zip(x, y)) + [(x[-1], 0)])
            verts.append(list(zip(x, y)))
            # verts = [[(-.2, 0), (-.2, 30), (0, 50), (0.2, 20), (0.2, 0)]]
            colors.append(np.array([0.4, 0, .9, (i+1)/limit]))

        poly = PolyCollection(verts, facecolors=colors)
        ax.add_collection3d(poly, zs=steps[-limit:], zdir='y')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(steps[-1], steps[-limit:][0])
        ax.set_zlim(y_min, y_max)

        # poly.set_alpha(0.7)
        

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

