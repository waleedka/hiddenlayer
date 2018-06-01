"""
watcher.py

Watcher class.

Written by Waleed Abdulla

Licensed under the MIT License
"""

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


