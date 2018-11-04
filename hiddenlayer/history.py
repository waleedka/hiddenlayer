"""
History class.

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
# Metric Class
###############################################################################

class Metric():
    """Represents the history of a single metric."""
    def __init__(self, history, name):
        self.name = name
        self.steps = history.steps
        self.data = np.array([history.history[s].get(name)
                              for s in self.steps])


###############################################################################
# History Class
###############################################################################

class History():
    """Tracks training progress and visualizes it.
    For example, use it to track the training and validation loss and accuracy
    and plot them.
    """
    
    def __init__(self):
        self.history = {}
        self.step = None
        self.epoch = None
        self.metrics = set()

    def log(self, step, **kwargs):
        """
        Okay to call multiple times for the same step.
        """
        # Update step
        step = str(step)  # TODO
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
        # TODO: cache the sorted steps for performance
        if not self.history:
            return []
        parts = next(iter(self.history.keys())).split(":")
        if len(parts) == 1:
            return list(map(str, sorted(map(int, self.history.keys()))))
        elif len(parts) == 2:
            steps = []
            for k in self.history.keys():
                s = k.split(":")
                steps.append( (int(s[0]), int(s[1])) )
            steps = sorted(steps)
            return ["{}:{}".format(e, b) for e, b in steps]

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
