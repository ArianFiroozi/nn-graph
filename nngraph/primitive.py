import torch
from enum import Enum
import onnx

class PrimitiveType(Enum):  
    MAC=1
    MAC2D=2
    INPUT=3
    OUTPUT=4
    ADD=5
    MUL=6
    SUB=7
    DIV=8
    MAX=9
    FLOOR=10                            
    RESHAPE=11
    MOD=12
    SHAPE=13
    SOFTMAX=14
    SLICE=15
    MIN=16
    TRANSPOSE=17
    CONCAT=18
    SQUEEZE=19
    UNSQUEEZE=20
    GATHER=21
    PADDING=22
    DILATION=23
    POOL=24

    UNKNOWN=0

class Primitive:
    def __init__(self, name: str, input_shape=None, output_shape=None, type: PrimitiveType = PrimitiveType.UNKNOWN, label:str="Prim"):
        self.name = name
        self.label = label
        self.type = type
        self.inputs = []
        self.outputs = []
        self.input_shape = None if input_shape is None else list(input_shape)
        self.output_shape = None if output_shape is None else list(output_shape)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Primitive) and self.name == other.name

    def __repr__(self):
        return self.label

    def get_label(self):
        return self.label

    def add_input_name(self, name: str):
        self.inputs.append(name)

    def get_name(self) -> str:
        return self.name

class InputPrim(Primitive):
    def __init__(self, name, shape, label:str="Input"):
        super().__init__(name, None, type=PrimitiveType.INPUT, label=label)
        self.shape=shape

    def get_label(self):
        return f"{self.label}\nshape: {self.shape}"

class OutputPrim(Primitive):
    def __init__(self, name, shape, label:str="Output"):
        super().__init__(name, None, type=PrimitiveType.INPUT, label=label)
        self.shape=shape

    def get_label(self):
        return f"{self.label}\nshape: {self.shape}"

class MacPrim(Primitive):
    def __init__(self, name, conv, label:str="Mac"):
        super().__init__(name, None, type=PrimitiveType.MAC, label=label)
        self.weight_indices=[None]
        if conv is not None:  # matmul mac is probably different from conv mac
            self.kernel_shape = conv.kernel_shape
            self.strides = conv.strides
            self.padding = conv.padding
            self.dilations = conv.dilations

    def get_label(self):
        return f"{self.label}"

class Mac2dPrim(Primitive):
    def __init__(self, name, conv, label:str="Mac2d"):
        super().__init__(name, None, type=PrimitiveType.MAC2D, label=label)
        self.kernel_shape = conv.kernel_shape
        self.strides = conv.strides
        self.padding = conv.padding
        self.dilations = conv.dilations
        self.weight_indices=[None]

    def get_label(self):
        return f"{self.label}"

class PoolPrim(Primitive):
    def __init__(self, name, pool, label:str="Pool"):
        super().__init__(name, None, type=PrimitiveType.POOL, label=label)
        self.kernel_shape = pool.kernel_shape
        self.strides = pool.strides
        self.padding = pool.padding
        self.dilations = pool.dilations

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}"

class DivPrim(Primitive):
    def __init__(self, name, label:str="Div"):
        super().__init__(name, None, type=PrimitiveType.DIV, label=label)

class AddPrim(Primitive):
    def __init__(self, name, label:str="Add"):
        super().__init__(name, None, type=PrimitiveType.ADD, label=label)

class SubPrim(Primitive):
    def __init__(self, name, label:str="Sub"):
        super().__init__(name, None, type=PrimitiveType.SUB, label=label)

class MulPrim(Primitive):
    def __init__(self, name, label:str="Mul"):
        super().__init__(name, None, type=PrimitiveType.MUL, label=label)

class FloorPrim(Primitive):
    def __init__(self, name, label:str="Floor"):
        super().__init__(name, None, type=PrimitiveType.FLOOR, label=label)

class MaxPrim(Primitive):
    def __init__(self, name, label:str="Max"):
        super().__init__(name, None, type=PrimitiveType.MAX, label=label)

class MinPrim(Primitive):
    def __init__(self, name, label:str="Min"):
        super().__init__(name, None, type=PrimitiveType.MIN, label=label)

class ReshapePrim(Primitive):
    def __init__(self, name, label:str="Reshape"):
        super().__init__(name, None, type=PrimitiveType.RESHAPE, label=label)

class ShapePrim(Primitive):
    def __init__(self, name, label:str="Shape"):
        super().__init__(name, None, type=PrimitiveType.SHAPE, label=label)

class ModPrim(Primitive):
    def __init__(self, name, label:str="Mod"):
        super().__init__(name, None, type=PrimitiveType.MOD, label=label)

class SoftMaxPrim(Primitive):
    def __init__(self, name, label:str="SoftMax"):
        super().__init__(name, None, type=PrimitiveType.SOFTMAX, label=label)

class SlicePrim(Primitive):
    def __init__(self, name, label:str="Slice"):
        super().__init__(name, None, type=PrimitiveType.SLICE, label=label)

class TransposePrim(Primitive):
    def __init__(self, name, label:str="Transpose"):
        super().__init__(name, None, type=PrimitiveType.TRANSPOSE, label=label)

class ConcatPrim(Primitive):
    def __init__(self, name, label:str="Concat"):
        super().__init__(name, None, type=PrimitiveType.CONCAT, label=label)

class SqueezePrim(Primitive):
    def __init__(self, name, label:str="Squeeze"):
        super().__init__(name, None, type=PrimitiveType.SQUEEZE, label=label)

class UnsqueezePrim(Primitive):
    def __init__(self, name, label:str="Unsqueeze"):
        super().__init__(name, None, type=PrimitiveType.UNSQUEEZE, label=label)

class GatherPrim(Primitive):
    def __init__(self, name, label:str="Gather"):
        super().__init__(name, None, type=PrimitiveType.GATHER, label=label)

class PaddingPrim(Primitive):
    def __init__(self, name, label:str="Padding"):
        super().__init__(name, None, type=PrimitiveType.PADDING, label=label)

class DilationPrim(Primitive):
    def __init__(self, name, label:str="Dilation"):
        super().__init__(name, None, type=PrimitiveType.DILATION, label=label)
