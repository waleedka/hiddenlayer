"""
HiddenLayer

Implementation of the Canvas class to render visualizations.

Written by Waleed Abdulla
Licensed under the MIT License
"""

import itertools
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython.display
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection


DEFAULT_THEME = {
    "fig_width": 12,  # inches
    "hist_outline_color": [0, 0, 0.9],
    "hist_color": [0.5, 0, 0.9],
}


def norm(image):
    """Normalize an image to [0, 1] range."""
    min_value = image.min()
    max_value = image.max()
    if min_value == max_value:
        return image - min_value
    return (image - min_value) / (max_value - min_value)


# TODO: Move inside Canvas and merge with draw_images
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
# Canvas Class
###############################################################################

class Canvas():

    def __init__(self):
        self._context = None
        self.theme = DEFAULT_THEME
        self.figure = None
        self.backend = matplotlib.get_backend()
        self.drawing_calls = []
        self.theme = DEFAULT_THEME

    def __enter__(self):
        self._context = "build"
        self.drawing_calls = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.render()

    def render(self):
        self._context = "run"
        # Clear output
        if 'inline' in self.backend:
            IPython.display.clear_output(wait=True)
            self.figure = None

        # Separate the draw_*() calls that generate a grid cell
        grid_calls = []
        silent_calls = []
        for c in self.drawing_calls:
            if c[0] == "draw_summary":
                silent_calls.append(c)
            else:
                grid_calls.append(c)

        # Header area
        # TODO: ideally, compute how much header area we need based on the
        #       length of text to show there. Right now, we're just using
        #       a fixed number multiplied by the number of calls. Since there
        #       is only one silent call, draw_summary(), then the header padding
        #       is either 0 or 0.1
        head_pad = 0.1 * len(silent_calls)

        width = self.theme['fig_width']
        if not self.figure:
            self.figure = plt.figure(figsize=(width, width/3 * (head_pad + len(grid_calls))))
        self.figure.clear()

        # Divide figure area by number of grid calls
        gs = matplotlib.gridspec.GridSpec(len(grid_calls), 1)

        # Call silent calls
        for c in silent_calls:
            getattr(self, c[0])(*c[1], **c[2])

        # Call grid methods
        for i, c in enumerate(grid_calls):
            method = c[0]
            # Create an axis for each call
            # Save in in self.ax so the drawing function has access to it
            self.ax = self.figure.add_subplot(gs[i])
            # Save the GridSpec as well
            self.gs = gs[i]
            # Call the method
            getattr(self, method)(*c[1], **c[2])
        # Cleanup after drawing
        self.ax = None
        self.gs = None
        gs.tight_layout(self.figure, rect=(0, 0, 1, 1-head_pad))

        # TODO: pause() allows the GUI to render but it's sluggish because it
        # only has 0.1 seconds of CPU time at each step. A better solution would be to
        # launch a separate process to render the GUI and pipe data to it.
        plt.pause(0.1)
        plt.show(block=False)
        self.drawing_calls = []
        self._context = None


    def __getattribute__(self, name):
        if name.startswith("draw_") and self._context != "run":
            def wrapper(*args, **kwargs):
                self.drawing_calls.append((name, args, kwargs))
                if not self._context:
                    self.render()
            return wrapper
        else:
            return object.__getattribute__(self, name)

    def save(self, file_name):
        self.figure.savefig(file_name)

    def draw_summary(self, history, title=""):
        """Inserts a text summary at the top that lists the number of steps and total
        training time."""
        # Generate summary string
        time_str = str(history.get_total_time()).split(".")[0]  # remove microseconds
        summary = "Step: {}      Time: {}".format(history.step, time_str)
        if title:
            summary = title + "\n\n" + summary
        self.figure.suptitle(summary)

    def draw_plot(self, metrics, labels=None, ylabel="", title=None):
        """
        metrics: One or more metrics parameters. Each represents the history
            of one metric.
        """
        metrics = metrics if isinstance(metrics, list) else [metrics]
        # Loop through metrics
        default_title = ""
        for i, m in enumerate(metrics):
            label = labels[i] if labels else m.name
            # TODO: use a standard formating function for values
            default_title += ("   " if default_title else "") + "{}: {}".format(label, m.data[-1])
            self.ax.plot(m.formatted_steps, m.data, label=label)
        title = default_title if title is None else title
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        self.ax.set_xlabel("Steps")
        self.ax.xaxis.set_major_locator(plt.AutoLocator())


    def draw_image(self, metric, limit=5):
        """Display a series of images at different time steps."""
        rows = 1
        cols = limit
        self.ax.axis("off")
        # Take the Axes gridspec and divide it into a grid
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=self.gs)
        # Loop through images in last few steps
        for i, image in enumerate(metric.data[-cols:]):
            ax = self.figure.add_subplot(gs[0, i])
            ax.axis('off')
            ax.set_title(metric.formatted_steps[-cols:][i])
            ax.imshow(norm(image))

    def draw_hist(self, metric, title=""):
        """Draw a series of histograms of the selected keys over different
        training steps.
        """
        # TODO: assert isinstance(list(values.values())[0], np.ndarray)

        rows = 1
        cols = 1
        limit = 10  # max steps to show

        # We need a 3D projection Subplot, so ignore the one provided to
        # as an create a new one.
        ax = self.figure.add_subplot(self.gs, projection="3d")
        ax.view_init(30, -80)

        # Compute histograms
        verts = []
        area_colors = []
        edge_colors = []
        for i, s in enumerate(metric.steps[-limit:]):
            hist, edges = np.histogram(metric.data[-i-1:])
            # X is bin centers
            x = np.diff(edges)/2 + edges[:-1]
            # Y is hist values
            y = hist
            x = np.concatenate([x[0:1], x, x[-1:]])
            y = np.concatenate([[0], y, [0]])

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

            alpha = 0.8 * (i+1) / min(limit, len(metric.steps))
            verts.append(list(zip(x, y)))
            area_colors.append(np.array(self.theme["hist_color"] + [alpha]))
            edge_colors.append(np.array(self.theme["hist_outline_color"] + [alpha]))

        poly = PolyCollection(verts, facecolors=area_colors, edgecolors=edge_colors)
        ax.add_collection3d(poly, zs=list(range(min(limit, len(metric.steps)))), zdir='y')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, limit)
        ax.set_yticklabels(metric.formatted_steps[-limit:])
        ax.set_zlim(y_min, y_max)
        ax.set_title(metric.name)

