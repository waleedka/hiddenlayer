"""
HiddenLayer

Utility functions.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

import numpy as np


###############################################################################
# Misc functions
###############################################################################

def to_data(value):
    """Standardize data types. Converts PyTorch tensors to Numpy arrays,
    and Numpy scalars to Python scalars."""
    # TODO: Use get_framework() for better detection.
    if value.__class__.__module__.startswith("torch"):
        import torch
        if isinstance(value, torch.nn.parameter.Parameter):
            value = value.data
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                value = value.detach()
            value = value.cpu().numpy().copy()
        # If 0-dim array, convert to scalar
        if not value.shape:
            value = value.item()
    # Convert Numpy scalar types to Python types
    if value.__class__.__module__ == "numpy" and value.__class__.__name__ != "ndarray":
        value = value.item()
    return value


def write(*args):
    """Like print(), but recognizes tensors and arrays and show
    more details about them.

    Example:
        hl.write("My Tensor", my_tensor)
    
        Prints:
            My Tensor  float32 (10, 3, 224, 224)  min: 0.0  max: 1.0
    """
    s = ""
    for a in args:
        # Convert tensors to Numpy arrays
        a = to_data(a)

        if isinstance(a, np.ndarray):
            # Numpy Array
            s += ("\t" if s else "") + "Tensor  {} {}  min: {:.3f}  max: {:.3f}".format(
                a.dtype, a.shape, a.min(), a.max())
            print(s)
            s = ""
        elif isinstance(a, list):
            s += ("\t" if s else "") + "list    len: {}  {}".format(len(a), a[:10])
        else:
            s += (" " if s else "") + str(a)
    if s:
        print(s)
