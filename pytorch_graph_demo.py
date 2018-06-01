"""
litegraph_pytorch.py

PyTorch graphs

Written by Waleed Abdulla

Licensed under the MIT License
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph import DirectedGraph
from layer import Layer

# Requires PyTorch 0.4
import torch
import torchvision.models
from distutils.version import LooseVersion
assert LooseVersion(torch.__version__) >= LooseVersion("0.4")

from builder_pytorch import build_pytorch_graph

#
# VGG 16
#

# VGG16 with BatchNorm
model = torchvision.models.vgg16_bn()

# Build graph
dg = build_pytorch_graph(model, torch.zeros([1, 3, 224, 224]), verbose=True)

# kind                      scopeName                                  inputs -> outputs
# onnx::Conv                VGG/Sequential[features]/Conv2d[0]         [0, 1, 2] -> [85]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[1]    [85, 3, 4, 5, 6] -> [86, 87, 88, 89, 90]
# onnx::Relu                VGG/Sequential[features]/ReLU[2]           [86] -> [91]
# onnx::Conv                VGG/Sequential[features]/Conv2d[3]         [91, 7, 8] -> [92]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[4]    [92, 9, 10, 11, 12] -> [93, 94, 95, 96, 97]
# onnx::Relu                VGG/Sequential[features]/ReLU[5]           [93] -> [98]
# onnx::MaxPool             VGG/Sequential[features]/MaxPool2d[6]      [98] -> [99]
# onnx::Conv                VGG/Sequential[features]/Conv2d[7]         [99, 13, 14] -> [100]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[8]    [100, 15, 16, 17, 18] -> [101, 102, 103, 104, 105]
# onnx::Relu                VGG/Sequential[features]/ReLU[9]           [101] -> [106]
# onnx::Conv                VGG/Sequential[features]/Conv2d[10]        [106, 19, 20] -> [107]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[11]   [107, 21, 22, 23, 24] -> [108, 109, 110, 111, 112]
# onnx::Relu                VGG/Sequential[features]/ReLU[12]          [108] -> [113]
# onnx::MaxPool             VGG/Sequential[features]/MaxPool2d[13]     [113] -> [114]
# onnx::Conv                VGG/Sequential[features]/Conv2d[14]        [114, 25, 26] -> [115]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[15]   [115, 27, 28, 29, 30] -> [116, 117, 118, 119, 120]
# onnx::Relu                VGG/Sequential[features]/ReLU[16]          [116] -> [121]
# onnx::Conv                VGG/Sequential[features]/Conv2d[17]        [121, 31, 32] -> [122]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[18]   [122, 33, 34, 35, 36] -> [123, 124, 125, 126, 127]
# onnx::Relu                VGG/Sequential[features]/ReLU[19]          [123] -> [128]
# onnx::Conv                VGG/Sequential[features]/Conv2d[20]        [128, 37, 38] -> [129]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[21]   [129, 39, 40, 41, 42] -> [130, 131, 132, 133, 134]
# onnx::Relu                VGG/Sequential[features]/ReLU[22]          [130] -> [135]
# onnx::MaxPool             VGG/Sequential[features]/MaxPool2d[23]     [135] -> [136]
# onnx::Conv                VGG/Sequential[features]/Conv2d[24]        [136, 43, 44] -> [137]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[25]   [137, 45, 46, 47, 48] -> [138, 139, 140, 141, 142]
# onnx::Relu                VGG/Sequential[features]/ReLU[26]          [138] -> [143]
# onnx::Conv                VGG/Sequential[features]/Conv2d[27]        [143, 49, 50] -> [144]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[28]   [144, 51, 52, 53, 54] -> [145, 146, 147, 148, 149]
# onnx::Relu                VGG/Sequential[features]/ReLU[29]          [145] -> [150]
# onnx::Conv                VGG/Sequential[features]/Conv2d[30]        [150, 55, 56] -> [151]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[31]   [151, 57, 58, 59, 60] -> [152, 153, 154, 155, 156]
# onnx::Relu                VGG/Sequential[features]/ReLU[32]          [152] -> [157]
# onnx::MaxPool             VGG/Sequential[features]/MaxPool2d[33]     [157] -> [158]
# onnx::Conv                VGG/Sequential[features]/Conv2d[34]        [158, 61, 62] -> [159]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[35]   [159, 63, 64, 65, 66] -> [160, 161, 162, 163, 164]
# onnx::Relu                VGG/Sequential[features]/ReLU[36]          [160] -> [165]
# onnx::Conv                VGG/Sequential[features]/Conv2d[37]        [165, 67, 68] -> [166]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[38]   [166, 69, 70, 71, 72] -> [167, 168, 169, 170, 171]
# onnx::Relu                VGG/Sequential[features]/ReLU[39]          [167] -> [172]
# onnx::Conv                VGG/Sequential[features]/Conv2d[40]        [172, 73, 74] -> [173]
# onnx::BatchNormalization  VGG/Sequential[features]/BatchNorm2d[41]   [173, 75, 76, 77, 78] -> [174, 175, 176, 177, 178]
# onnx::Relu                VGG/Sequential[features]/ReLU[42]          [174] -> [179]
# onnx::MaxPool             VGG/Sequential[features]/MaxPool2d[43]     [179] -> [180]
# onnx::Flatten             VGG                                        [180] -> [181]
# onnx::Gemm                VGG/Sequential[classifier]/Linear[0]       [181, 79, 80] -> [182]
# onnx::Relu                VGG/Sequential[classifier]/ReLU[1]         [182] -> [183]
# onnx::Dropout             VGG/Sequential[classifier]/Dropout[2]      [183] -> [184, 185]
# onnx::Gemm                VGG/Sequential[classifier]/Linear[3]       [184, 81, 82] -> [186]
# onnx::Relu                VGG/Sequential[classifier]/ReLU[4]         [186] -> [187]
# onnx::Dropout             VGG/Sequential[classifier]/Dropout[5]      [187] -> [188, 189]
# onnx::Gemm                VGG/Sequential[classifier]/Linear[6]       [188, 83, 84] -> [190]

# Draw graph
dg.draw_graph(simplify=True, output_shapes=False, verbose=True)

# Replacing sequence [/<Layer: op: linear         , name: Linear         , id: VGG/Sequential[classifier]/Linear[0]/outputs/182  , title: Linear         , repeat:  1, params: {'alpha': 1.0, 'beta': 1.0, 'broadcast': 1, 'transB': 1}>/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[classifier]/ReLU[1]/outputs/183    , title: Relu           , repeat:  1>/<Layer: op: dropout        , name: Dropout        , id: VGG/Sequential[classifier]/Dropout[2]/outputs/184/185, title: Dropout        , repeat:  1, params: {'is_test': 0, 'ratio': 0.5}                      >] with combo <Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                               16659341645538463734, title: Linear/Relu/Dropout, repeat:  1>
# Replacing sequence [/<Layer: op: linear         , name: Linear         , id: VGG/Sequential[classifier]/Linear[3]/outputs/186  , title: Linear         , repeat:  1, params: {'alpha': 1.0, 'beta': 1.0, 'broadcast': 1, 'transB': 1}>/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[classifier]/ReLU[4]/outputs/187    , title: Relu           , repeat:  1>/<Layer: op: dropout        , name: Dropout        , id: VGG/Sequential[classifier]/Dropout[5]/outputs/188/189, title: Dropout        , repeat:  1, params: {'is_test': 0, 'ratio': 0.5}                      >] with combo <Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                                 183353858164065529, title: Linear/Relu/Dropout, repeat:  1>
# Replacing sequence [/<Layer: op: linear         , name: Linear         , id: VGG/Sequential[classifier]/Linear[6]/outputs/190  , title: Linear         , repeat:  1, params: {'alpha': 1.0, 'beta': 1.0, 'broadcast': 1, 'transB': 1}>] with combo <Layer: op: linear/relu/dropout, name: Linear         , id:                                9521916156652872523, title: Linear         , repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[0]/outputs/85     , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[1]/outputs/86/87/88/batch_norm_dead_output-89/batch_norm_dead_output-90, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[2]/outputs/91       , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               14844735543913119523, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[3]/outputs/92     , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[4]/outputs/93/94/95/batch_norm_dead_output-96/batch_norm_dead_output-97, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[5]/outputs/98       , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                8986409883394785310, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[7]/outputs/100    , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[8]/outputs/101/102/103/batch_norm_dead_output-104/batch_norm_dead_output-105, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[9]/outputs/106      , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                1976162069957399864, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[10]/outputs/107   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[11]/outputs/108/109/110/batch_norm_dead_output-111/batch_norm_dead_output-112, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[12]/outputs/113     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                 521901217502697038, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[14]/outputs/115   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[15]/outputs/116/117/118/batch_norm_dead_output-119/batch_norm_dead_output-120, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[16]/outputs/121     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                6615185624271265518, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[17]/outputs/122   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[18]/outputs/123/124/125/batch_norm_dead_output-126/batch_norm_dead_output-127, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[19]/outputs/128     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               10367349716046967604, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[20]/outputs/129   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[21]/outputs/130/131/132/batch_norm_dead_output-133/batch_norm_dead_output-134, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[22]/outputs/135     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               11597376757146529840, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[24]/outputs/137   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[25]/outputs/138/139/140/batch_norm_dead_output-141/batch_norm_dead_output-142, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[26]/outputs/143     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                6390932437100149525, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[27]/outputs/144   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[28]/outputs/145/146/147/batch_norm_dead_output-148/batch_norm_dead_output-149, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[29]/outputs/150     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               10273622060295509752, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[30]/outputs/151   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[31]/outputs/152/153/154/batch_norm_dead_output-155/batch_norm_dead_output-156, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[32]/outputs/157     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               12601111427079161177, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[34]/outputs/159   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[35]/outputs/160/161/162/batch_norm_dead_output-163/batch_norm_dead_output-164, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[36]/outputs/165     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                 758059361734585284, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[37]/outputs/166   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[38]/outputs/167/168/169/batch_norm_dead_output-170/batch_norm_dead_output-171, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[39]/outputs/172     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                5008926413156358678, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [/<Layer: op: conv           , name: Conv           , id: VGG/Sequential[features]/Conv2d[40]/outputs/173   , title: Conv3x3        , repeat:  1, params: {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}>/<Layer: op: bn             , name: BatchNorm      , id: VGG/Sequential[features]/BatchNorm2d[41]/outputs/174/175/176/batch_norm_dead_output-177/batch_norm_dead_output-178, title: BatchNorm      , repeat:  1, params: {'epsilon': 1e-05, 'is_test': 0, 'momentum': 0.9} >/<Layer: op: relu           , name: Relu           , id: VGG/Sequential[features]/ReLU[42]/outputs/179     , title: Relu           , repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                8052854920043721786, title: Conv3x3/BatchNorm/Relu, repeat:  1>
# Replacing sequence [<Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                               16659341645538463734, title: Linear/Relu/Dropout, repeat:  1>/<Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                                 183353858164065529, title: Linear/Relu/Dropout, repeat:  1>] with combo <Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                                1664777053450313182, title: Linear/Relu/Dropout, repeat:  2>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               14844735543913119523, title: Conv3x3/BatchNorm/Relu, repeat:  1>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                8986409883394785310, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                 317753656148653240, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                1976162069957399864, title: Conv3x3/BatchNorm/Relu, repeat:  1>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                 521901217502697038, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               16928034434214896614, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                6615185624271265518, title: Conv3x3/BatchNorm/Relu, repeat:  1>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               10367349716046967604, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               11576454374829693928, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                6390932437100149525, title: Conv3x3/BatchNorm/Relu, repeat:  1>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               10273622060295509752, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                7075296053023081877, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                 758059361734585284, title: Conv3x3/BatchNorm/Relu, repeat:  1>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                5008926413156358678, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                9387480097963704518, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# Replacing sequence [<Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                                1664777053450313182, title: Linear/Relu/Dropout, repeat:  2>/<Layer: op: linear/relu/dropout, name: Linear         , id:                                9521916156652872523, title: Linear         , repeat:  1>] with combo <Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                                3555284668127683616, title: Linear/Relu/Dropout, repeat:  3>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               11576454374829693928, title: Conv3x3/BatchNorm/Relu, repeat:  2>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               11597376757146529840, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               11415110985134755736, title: Conv3x3/BatchNorm/Relu, repeat:  3>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                7075296053023081877, title: Conv3x3/BatchNorm/Relu, repeat:  2>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               12601111427079161177, title: Conv3x3/BatchNorm/Relu, repeat:  1>] with combo <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                2193678720638173042, title: Conv3x3/BatchNorm/Relu, repeat:  3>
# Replacing sequence [<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                9387480097963704518, title: Conv3x3/BatchNorm/Relu, repeat:  2>/<Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:

# List layers in the graph
print("Layers:")
dg.list_layers()

# <Layer: op: maxpool        , name: MaxPool        , id: VGG/Sequential[features]/MaxPool2d[6]/outputs/99  , title: MaxPool2x2     , repeat:  1, params: {'kernel_shape': [2, 2], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}>
# <Layer: op: maxpool        , name: MaxPool        , id: VGG/Sequential[features]/MaxPool2d[13]/outputs/114, title: MaxPool2x2     , repeat:  1, params: {'kernel_shape': [2, 2], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}>
# <Layer: op: maxpool        , name: MaxPool        , id: VGG/Sequential[features]/MaxPool2d[23]/outputs/136, title: MaxPool2x2     , repeat:  1, params: {'kernel_shape': [2, 2], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}>
# <Layer: op: maxpool        , name: MaxPool        , id: VGG/Sequential[features]/MaxPool2d[33]/outputs/158, title: MaxPool2x2     , repeat:  1, params: {'kernel_shape': [2, 2], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}>
# <Layer: op: maxpool        , name: MaxPool        , id: VGG/Sequential[features]/MaxPool2d[43]/outputs/180, title: MaxPool2x2     , repeat:  1, params: {'kernel_shape': [2, 2], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}>
# <Layer: op: Flatten        , name: Flatten        , id: VGG/outputs/181                                   , title: Flatten        , repeat:  1, params: {'axis': 1}                                       >
# <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                 317753656148653240, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               16928034434214896614, title: Conv3x3/BatchNorm/Relu, repeat:  2>
# <Layer: op: linear/relu/dropout, name: Linear/Relu/Dropout, id:                                3555284668127683616, title: Linear/Relu/Dropout, repeat:  3>
# <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               11415110985134755736, title: Conv3x3/BatchNorm/Relu, repeat:  3>
# <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                                2193678720638173042, title: Conv3x3/BatchNorm/Relu, repeat:  3>
# <Layer: op: conv/bn/relu   , name: Conv3x3/BatchNorm/Relu, id:                               13900048100721660265, title: Conv3x3/BatchNorm/Relu, repeat:  3>

#
# ResNet
#

# Resnet50
model = torchvision.models.resnet50()

# Build graph
dg = build_pytorch_graph(model, torch.zeros([1, 3, 224, 224]), verbose=False)

# Draw graph
dg.draw_graph(simplify=True, output_shapes=False, verbose=False)

# List layers in the graph
print("Layers:")
dg.list_layers()

# Layers:
# <Layer: op: maxpool        , name: MaxPool        , id: ResNet/MaxPool2d[maxpool]/outputs/275             , title: MaxPool3x3     , repeat:  1, params: {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer1]/Bottleneck[0]/outputs/302, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]/outputs/303, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer1]/Bottleneck[1]/outputs/324, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]/outputs/325, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer1]/Bottleneck[2]/outputs/346, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]/outputs/347, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer2]/Bottleneck[0]/outputs/374, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]/outputs/375, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer2]/Bottleneck[1]/outputs/396, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]/outputs/397, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer2]/Bottleneck[2]/outputs/418, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]/outputs/419, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer2]/Bottleneck[3]/outputs/440, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]/outputs/441, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer3]/Bottleneck[0]/outputs/468, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]/outputs/469, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer3]/Bottleneck[1]/outputs/490, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]/outputs/491, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer3]/Bottleneck[2]/outputs/512, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]/outputs/513, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer3]/Bottleneck[3]/outputs/534, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]/outputs/535, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer3]/Bottleneck[4]/outputs/556, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]/outputs/557, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer3]/Bottleneck[5]/outputs/578, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]/outputs/579, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer4]/Bottleneck[0]/outputs/606, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]/outputs/607, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer4]/Bottleneck[1]/outputs/628, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]/outputs/629, title: Relu           , repeat:  1>
# <Layer: op: Add            , name: +              , id: ResNet/Sequential[layer4]/Bottleneck[2]/outputs/650, title: +              , repeat:  1>
# <Layer: op: relu           , name: Relu           , id: ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]/outputs/651, title: Relu           , repeat:  1>
# <Layer: op: AveragePool    , name: AveragePool    , id: ResNet/AvgPool2d[avgpool]/outputs/652             , title: AveragePool7x7 , repeat:  1, params: {'kernel_shape': [7, 7], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}>
# <Layer: op: Flatten        , name: Flatten        , id: ResNet/outputs/653                                , title: Flatten        , repeat:  1, params: {'axis': 1}                                       >
# <Layer: op: linear/relu/dropout, name: Linear         , id:                                1271038422604150267, title: Linear         , repeat:  1>
# <Layer: op: conv/bn/relu   , name: Conv7x7/BatchNorm/Relu, id:                                4431989414349523760, title: Conv7x7/BatchNorm/Relu, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                7844379451683829469, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               18356494911128454375, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                2204997981412691783, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                 378303124276004442, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                1399401227915899574, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                1322468764620712966, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                8002611444133218313, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               13073541897307003370, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               13670213883941035993, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               14979207121606756297, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                3252867688741610706, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                3282870041304493567, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               17615453743999709953, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                 192737139652763165, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                3037757811020627142, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                1543069523610900003, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               12744800535569446849, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               15353959286481148782, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                               14657893098943659710, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn        , name: Conv1x1/BatchNorm, id:                                7392169211644370732, title: Conv1x1/BatchNorm, repeat:  1>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                                9938797332929817182, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               14016247944610688953, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                                7232174824052228992, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               17501548392626711613, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                                2034756518805251309, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               17954787308661930625, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                                1449491880091086382, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               15849218031646893817, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               17709365859937739882, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               14218936257607903561, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                                7798823097230203322, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               17773302515033272161, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               10461919906088428733, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               17037065461751268179, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                                6689815151965924244, title: Conv1x1/BatchNorm/Relu, repeat:  2>
# <Layer: op: conv/bn/relu   , name: Conv1x1/BatchNorm/Relu, id:                               10258069244027751964, title: Conv1x1/BatchNorm/Relu, repeat:  2>

#
# Trace
#

# Experiments with Pytorch traces to extract graph details.
args = torch.zeros([1, 3, 224, 224])
trace, out = torch.jit.get_trace_graph(model, args)
torch.onnx._optimize_trace(trace, False)
graph = trace.graph()

f = "{:25} {:40}   {} -> {}"
print(f.format("kind", "scopeName", "inputs", "outputs"))
for node in graph.nodes():
    print(f.format(node.kind(), node.scopeName(),
                   [i.unique() for i in node.inputs()],
                   [i.unique() for i in node.outputs()]
                   ))

# kind                      scopeName                                  inputs -> outputs
# onnx::Conv                ResNet/Conv2d[conv1]                       [0, 1] -> [268]
# onnx::BatchNormalization  ResNet/BatchNorm2d[bn1]                    [268, 2, 3, 4, 5] -> [269, 270, 271, 272, 273]
# onnx::Relu                ResNet/ReLU[relu]                          [269] -> [274]
# onnx::MaxPool             ResNet/MaxPool2d[maxpool]                  [274] -> [275]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]   [275, 6] -> [276]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]   [276, 7, 8, 9, 10] -> [277, 278, 279, 280, 281]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]   [277] -> [282]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]   [282, 11] -> [283]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]   [283, 12, 13, 14, 15] -> [284, 285, 286, 287, 288]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]   [284] -> [289]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]   [289, 16] -> [290]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]   [290, 17, 18, 19, 20] -> [291, 292, 293, 294, 295]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]   [275, 21] -> [296]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]   [296, 22, 23, 24, 25] -> [297, 298, 299, 300, 301]
# onnx::Add                 ResNet/Sequential[layer1]/Bottleneck[0]    [291, 297] -> [302]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]   [302] -> [303]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]   [303, 26] -> [304]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]   [304, 27, 28, 29, 30] -> [305, 306, 307, 308, 309]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]   [305] -> [310]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]   [310, 31] -> [311]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]   [311, 32, 33, 34, 35] -> [312, 313, 314, 315, 316]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]   [312] -> [317]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]   [317, 36] -> [318]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]   [318, 37, 38, 39, 40] -> [319, 320, 321, 322, 323]
# onnx::Add                 ResNet/Sequential[layer1]/Bottleneck[1]    [319, 303] -> [324]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]   [324] -> [325]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]   [325, 41] -> [326]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]   [326, 42, 43, 44, 45] -> [327, 328, 329, 330, 331]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]   [327] -> [332]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]   [332, 46] -> [333]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]   [333, 47, 48, 49, 50] -> [334, 335, 336, 337, 338]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]   [334] -> [339]
# onnx::Conv                ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]   [339, 51] -> [340]
# onnx::BatchNormalization  ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]   [340, 52, 53, 54, 55] -> [341, 342, 343, 344, 345]
# onnx::Add                 ResNet/Sequential[layer1]/Bottleneck[2]    [341, 325] -> [346]
# onnx::Relu                ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]   [346] -> [347]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]   [347, 56] -> [348]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]   [348, 57, 58, 59, 60] -> [349, 350, 351, 352, 353]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]   [349] -> [354]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]   [354, 61] -> [355]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]   [355, 62, 63, 64, 65] -> [356, 357, 358, 359, 360]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]   [356] -> [361]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]   [361, 66] -> [362]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]   [362, 67, 68, 69, 70] -> [363, 364, 365, 366, 367]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]   [347, 71] -> [368]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]   [368, 72, 73, 74, 75] -> [369, 370, 371, 372, 373]
# onnx::Add                 ResNet/Sequential[layer2]/Bottleneck[0]    [363, 369] -> [374]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]   [374] -> [375]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]   [375, 76] -> [376]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]   [376, 77, 78, 79, 80] -> [377, 378, 379, 380, 381]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]   [377] -> [382]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]   [382, 81] -> [383]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]   [383, 82, 83, 84, 85] -> [384, 385, 386, 387, 388]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]   [384] -> [389]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]   [389, 86] -> [390]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]   [390, 87, 88, 89, 90] -> [391, 392, 393, 394, 395]
# onnx::Add                 ResNet/Sequential[layer2]/Bottleneck[1]    [391, 375] -> [396]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]   [396] -> [397]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]   [397, 91] -> [398]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]   [398, 92, 93, 94, 95] -> [399, 400, 401, 402, 403]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]   [399] -> [404]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]   [404, 96] -> [405]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]   [405, 97, 98, 99, 100] -> [406, 407, 408, 409, 410]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]   [406] -> [411]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]   [411, 101] -> [412]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]   [412, 102, 103, 104, 105] -> [413, 414, 415, 416, 417]
# onnx::Add                 ResNet/Sequential[layer2]/Bottleneck[2]    [413, 397] -> [418]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]   [418] -> [419]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]   [419, 106] -> [420]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]   [420, 107, 108, 109, 110] -> [421, 422, 423, 424, 425]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]   [421] -> [426]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]   [426, 111] -> [427]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]   [427, 112, 113, 114, 115] -> [428, 429, 430, 431, 432]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]   [428] -> [433]
# onnx::Conv                ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]   [433, 116] -> [434]
# onnx::BatchNormalization  ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]   [434, 117, 118, 119, 120] -> [435, 436, 437, 438, 439]
# onnx::Add                 ResNet/Sequential[layer2]/Bottleneck[3]    [435, 419] -> [440]
# onnx::Relu                ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]   [440] -> [441]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]   [441, 121] -> [442]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]   [442, 122, 123, 124, 125] -> [443, 444, 445, 446, 447]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]   [443] -> [448]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]   [448, 126] -> [449]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]   [449, 127, 128, 129, 130] -> [450, 451, 452, 453, 454]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]   [450] -> [455]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]   [455, 131] -> [456]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]   [456, 132, 133, 134, 135] -> [457, 458, 459, 460, 461]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]   [441, 136] -> [462]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]   [462, 137, 138, 139, 140] -> [463, 464, 465, 466, 467]
# onnx::Add                 ResNet/Sequential[layer3]/Bottleneck[0]    [457, 463] -> [468]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]   [468] -> [469]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]   [469, 141] -> [470]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]   [470, 142, 143, 144, 145] -> [471, 472, 473, 474, 475]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]   [471] -> [476]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]   [476, 146] -> [477]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]   [477, 147, 148, 149, 150] -> [478, 479, 480, 481, 482]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]   [478] -> [483]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]   [483, 151] -> [484]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]   [484, 152, 153, 154, 155] -> [485, 486, 487, 488, 489]
# onnx::Add                 ResNet/Sequential[layer3]/Bottleneck[1]    [485, 469] -> [490]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]   [490] -> [491]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]   [491, 156] -> [492]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]   [492, 157, 158, 159, 160] -> [493, 494, 495, 496, 497]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]   [493] -> [498]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]   [498, 161] -> [499]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]   [499, 162, 163, 164, 165] -> [500, 501, 502, 503, 504]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]   [500] -> [505]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]   [505, 166] -> [506]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]   [506, 167, 168, 169, 170] -> [507, 508, 509, 510, 511]
# onnx::Add                 ResNet/Sequential[layer3]/Bottleneck[2]    [507, 491] -> [512]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]   [512] -> [513]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]   [513, 171] -> [514]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]   [514, 172, 173, 174, 175] -> [515, 516, 517, 518, 519]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]   [515] -> [520]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]   [520, 176] -> [521]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]   [521, 177, 178, 179, 180] -> [522, 523, 524, 525, 526]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]   [522] -> [527]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]   [527, 181] -> [528]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]   [528, 182, 183, 184, 185] -> [529, 530, 531, 532, 533]
# onnx::Add                 ResNet/Sequential[layer3]/Bottleneck[3]    [529, 513] -> [534]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]   [534] -> [535]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]   [535, 186] -> [536]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]   [536, 187, 188, 189, 190] -> [537, 538, 539, 540, 541]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]   [537] -> [542]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]   [542, 191] -> [543]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]   [543, 192, 193, 194, 195] -> [544, 545, 546, 547, 548]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]   [544] -> [549]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]   [549, 196] -> [550]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]   [550, 197, 198, 199, 200] -> [551, 552, 553, 554, 555]
# onnx::Add                 ResNet/Sequential[layer3]/Bottleneck[4]    [551, 535] -> [556]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]   [556] -> [557]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]   [557, 201] -> [558]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]   [558, 202, 203, 204, 205] -> [559, 560, 561, 562, 563]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]   [559] -> [564]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]   [564, 206] -> [565]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]   [565, 207, 208, 209, 210] -> [566, 567, 568, 569, 570]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]   [566] -> [571]
# onnx::Conv                ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]   [571, 211] -> [572]
# onnx::BatchNormalization  ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]   [572, 212, 213, 214, 215] -> [573, 574, 575, 576, 577]
# onnx::Add                 ResNet/Sequential[layer3]/Bottleneck[5]    [573, 557] -> [578]
# onnx::Relu                ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]   [578] -> [579]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]   [579, 216] -> [580]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]   [580, 217, 218, 219, 220] -> [581, 582, 583, 584, 585]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]   [581] -> [586]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]   [586, 221] -> [587]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]   [587, 222, 223, 224, 225] -> [588, 589, 590, 591, 592]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]   [588] -> [593]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]   [593, 226] -> [594]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]   [594, 227, 228, 229, 230] -> [595, 596, 597, 598, 599]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]   [579, 231] -> [600]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]   [600, 232, 233, 234, 235] -> [601, 602, 603, 604, 605]
# onnx::Add                 ResNet/Sequential[layer4]/Bottleneck[0]    [595, 601] -> [606]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]   [606] -> [607]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]   [607, 236] -> [608]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]   [608, 237, 238, 239, 240] -> [609, 610, 611, 612, 613]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]   [609] -> [614]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]   [614, 241] -> [615]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]   [615, 242, 243, 244, 245] -> [616, 617, 618, 619, 620]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]   [616] -> [621]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]   [621, 246] -> [622]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]   [622, 247, 248, 249, 250] -> [623, 624, 625, 626, 627]
# onnx::Add                 ResNet/Sequential[layer4]/Bottleneck[1]    [623, 607] -> [628]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]   [628] -> [629]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]   [629, 251] -> [630]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]   [630, 252, 253, 254, 255] -> [631, 632, 633, 634, 635]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]   [631] -> [636]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]   [636, 256] -> [637]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]   [637, 257, 258, 259, 260] -> [638, 639, 640, 641, 642]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]   [638] -> [643]
# onnx::Conv                ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]   [643, 261] -> [644]
# onnx::BatchNormalization  ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]   [644, 262, 263, 264, 265] -> [645, 646, 647, 648, 649]
# onnx::Add                 ResNet/Sequential[layer4]/Bottleneck[2]    [645, 629] -> [650]
# onnx::Relu                ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]   [650] -> [651]
# onnx::AveragePool         ResNet/AvgPool2d[avgpool]                  [651] -> [652]
# onnx::Flatten             ResNet                                     [652] -> [653]
# onnx::Gemm                ResNet/Linear[fc]                          [653, 266, 267] -> [654]

for node in list(graph.nodes())[:10]:
    print(node)

# %268 : Float(1, 64, 112, 112) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[7, 7], pads=[3, 3, 3, 3], strides=[2, 2]](%0, %1), scope: ResNet/Conv2d[conv1]
#
# %269 : Float(1, 64, 112, 112), %270 : Float(64), %271 : Float(64), %batch_norm_dead_output-273 : Dynamic, %batch_norm_dead_output-274 : Dynamic = onnx::BatchNormalization[epsilon=1e-05, is_test=0, momentum=0.9](%268, %2, %3, %4, %5), scope: ResNet/BatchNorm2d[bn1]
#
# %274 : Float(1, 64, 112, 112) = onnx::Relu(%269), scope: ResNet/ReLU[relu]
#
# %275 : Float(1, 64, 56, 56) = onnx::MaxPool[kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]](%274), scope: ResNet/MaxPool2d[maxpool]
#
# %276 : Float(1, 64, 56, 56) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]](%275, %6), scope: ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]
#
# %277 : Float(1, 64, 56, 56), %278 : Float(64), %279 : Float(64), %batch_norm_dead_output-282 : Dynamic, %batch_norm_dead_output-283 : Dynamic = onnx::BatchNormalization[epsilon=1e-05, is_test=0, momentum=0.9](%276, %7, %8, %9, %10), scope: ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]
#
# %282 : Float(1, 64, 56, 56) = onnx::Relu(%277), scope: ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]
#
# %283 : Float(1, 64, 56, 56) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%282, %11), scope: ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]
#
# %284 : Float(1, 64, 56, 56), %285 : Float(64), %286 : Float(64), %batch_norm_dead_output-290 : Dynamic, %batch_norm_dead_output-291 : Dynamic = onnx::BatchNormalization[epsilon=1e-05, is_test=0, momentum=0.9](%283, %12, %13, %14, %15), scope: ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]
#
# %289 : Float(1, 64, 56, 56) = onnx::Relu(%284), scope: ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]
