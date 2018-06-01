# Import methods to expose in the library
# BUGBUG
# The imports below are problematic:
# What if I have TF but not PyTorch (or vice versa) installed in my current virtualenv?
# Then, one of the two next import will fail and the library won't load, as shown here:
#   File "E:/repos/litegraph-dev/tf_demo.py", line 35, in <module>
#     from ww import build_tf_graph
#   File "E:\repos\litegraph-dev\ww\__init__.py", line 2, in <module>
#     from .builder_pytorch import build_pytorch_graph
#   File "E:\repos\litegraph-dev\ww\builder_pytorch.py", line 18, in <module>
#     import torch
# ModuleNotFoundError: No module named 'torch'
# Is it really fair to require that TF **and** PyTorch both be installed before I can use ww?
# from .builder_pytorch import build_pytorch_graph
# from .builder_tf import build_tf_graph
# from .watcher import show
# from .watcher import show_images
# from .watcher import Watcher
