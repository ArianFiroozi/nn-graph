import torch
from enum import Enum
import onnx

class PrimitiveType(Enum):
    MAC=1
    MAC2D=2
    UNKNOWN=0

class Primitive:
    def __init__(self, name: str, input_shape=None, output_shape=None, type: PrimitiveType = PrimitiveType.UNKNOWN, label: str = "Prim"):
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
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}"

    def add_input_name(self, name: str):
        self.inputs.append(name)

    def get_name(self) -> str:
        return self.name

class MacPrim(Primitive):
    def __init__(self, name, conv, label: str = "MacPrim"):
        super().__init__(name, None, type=PrimitiveType.MAC, label=label)

        if conv is not None:  # matmul mac is probably different from conv mac
            self.kernel_shape = conv.kernel_shape
            self.strides = conv.strides
            self.padding = conv.padding
            self.dilations = conv.dilations

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}"

class Mac2dPrim(Primitive):
    def __init__(self, name, conv, label: str = "Mac2dPrim"):
        super().__init__(name, None, type=PrimitiveType.MAC2D, label=label)
        self.kernel_shape = conv.kernel_shape
        self.strides = conv.strides
        self.padding = conv.padding
        self.dilations = conv.dilations

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}"

class DivPrim(Primitive):
    def __init__(self, name, conv, label: str = "Mac2dPrim"):
        super().__init__(name, None, type=PrimitiveType.MAC2D, label=label)
        self.kernel_shape = conv.kernel_shape
        self.strides = conv.strides
        self.padding = conv.padding
        self.dilations = conv.dilations

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}"

