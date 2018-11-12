# Set matplotlib backend to Agg
# *MUST* be done BEFORE importing hiddenlayer or libs that import matplotlib
import matplotlib
matplotlib.use("Agg")

import os
import time
import random
import numpy as np
import hiddenlayer as hl

# Create output directory in project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "demo_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# A History object to store metrics
h = hl.History()

# A Canvas object to draw the metrics
c = hl.Canvas()

# Simulate a training loop with two metrics: loss and accuracy
loss = 1
accuracy = 0
for step in range(1000):
    # Fake loss and accuracy
    loss -= loss * np.random.uniform(-.09, 0.1)
    accuracy += (1 - accuracy) * np.random.uniform(-.09, 0.1)

    # Log metrics and display them at certain intervals
    if step % 10 == 0:
        # Store metrics in the history object
        h.log(step, loss=loss, accuracy=accuracy)

        # Print progress status
        h.progress()

        # Less occasionally, save a snapshot of the graphs
        if step % 100 == 0:
            # Plot the two metrics in one graph
            c.draw_plot([h["loss"], h["accuracy"]])
            # Save the canvas
            c.save(os.path.join(OUTPUT_DIR, "training_progress.png"))

            # You can also save the history to a file to load and inspect layer
            h.save(os.path.join(OUTPUT_DIR, "training_progress.pkl"))

        time.sleep(0.1)
