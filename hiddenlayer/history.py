"""
HiddenLayer

Implementation of the History class to train training metrics.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

import math
import random
import io
import itertools
import time
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from . import utils


###############################################################################
# Helper Functions
###############################################################################

def format_step(step, zero_prefix=False):
    """Return the step value in format suitable for display."""
    if isinstance(step, int):
        return "{:06}".format(step) if zero_prefix else "{}".format(step)
    elif isinstance(step, tuple):
        return "{:04}:{:06}".format(*step) if zero_prefix else "{}:{}".format(*step)


###############################################################################
# Metric Class
###############################################################################

class Metric():
    """Represents the history of a single metric."""
    def __init__(self, history, name):
        self.name = name
        self.steps = history.steps
        self.data = np.array([history.history[s].get(name)
                              for s in self.steps])

    @property
    def formatted_steps(self):
        return [format_step(s) for s in self.steps]


###############################################################################
# History Class
###############################################################################

class History():
    """Tracks training progress and visualizes it.
    For example, use it to track the training and validation loss and accuracy
    and plot them.
    """
    
    def __init__(self):
        self.step = None  # Last reported step
        self.metrics = set()  # Names of all metrics reported so far
        self.history = {}  # Dict of steps and metrics {step: [metrics...]}

    def log(self, step, **kwargs):
        """Record metrics at a specific step. E.g.

        my_history.log(34, loss=2.3, accuracy=0.2)

        Okay to call multiple times for the same step. New values overwrite
        older ones if they have the same metric name.

        step: An integer or tuple of integers. If a tuple, then the first
            value is considered to be the epoch and the second is the step
            within the epoch.
        """
        assert isinstance(step, (int, tuple)), "Step must be an int or a tuple of two ints"
        self.step = step
        # Any new metrics we haven't seen before?
        self.metrics |= set(kwargs.keys())
        # Insert (or update) record of the step
        if step not in self.history:
            self.history[step] = {}
        self.history[step].update({k:utils.to_data(v) for k, v in kwargs.items()})
        # Update step timestamp
        self.history[step]["__timestamp__"] = time.time()

    @property
    def steps(self):
        """Returns a list of all steps logged so far. Guaranteed to be
        sorted correctly."""
        if not self.history:
            return []
        # TODO: Consider caching the sorted steps for performance
        return sorted(self.history.keys())

    @property
    def formatted_steps(self):
        return [format_step(s) for s in self.steps]

    def __getitem__(self, metric):
        return Metric(self, metric)

    def progress(self):
        # TODO: Erase the previous progress text to update in place
        text = "Step {}: ".format(self.step)
        metrics = self.history[self.step]
        for k, v in metrics.items():
            # Skip timestamp
            if k == "__timestamp__":
                continue
            # Exclude lists, dicts, and arrays
            # TODO: ideally, include the skipped types with a compact representation
            if not isinstance(v, (list, dict, np.ndarray)):
                text += "{}: {}  ".format(k, v)
        print(text)

    def summary(self):
        # TODO: Include more details in the summary
        print("Last Step: {}".format(self.step))
        print("Training Time: {}".format(self.get_total_time()))

    def get_total_time(self):
        """Returns the total period between when the first and last steps
        where logged. This usually correspnods to the total training time
        if there were no gaps in the training.
        """
        first_step = self.steps[0]
        last_step = self.steps[-1]
        seconds = self.history[last_step]["__timestamp__"] \
                  - self.history[first_step]["__timestamp__"]
        return datetime.timedelta(seconds=seconds)

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.history, f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.history = pickle.load(f)
        # Set last step and metrics
        self.step = self.steps[-1]
        unique_metrics = set(itertools.chain(*[m.keys() for m in self.history.values()]))
        self.metrics = unique_metrics - {"__timestamp__",}
