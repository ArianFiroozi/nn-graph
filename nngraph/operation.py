import torch
from graphviz import Digraph
import pickle
import torch.nn as nn
from enum import Enum
import onnx
import onnx2torch

class OperationType(Enum):
    INPUT=1
    MAC=2
    ADD=3
    MULT=4
    OUTPUT=5
    POOL=6
    PADD=7
    DILATION=8
    CONST=9
    PROJECT=10
    DOT_PRODUCT=11
    UNKNOWN=0

class Operation:
    def __init__(self, name:str, node:onnx.NodeProto, type:OperationType=0, label:str="OP"):
        self.name=name # unique
        self.label=label
        self.type=type

        if node is not None:
            self.inputs=node.input
            self.outputs=node.output
        else:
            self.inputs=[]
            self.outputs=[]
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Operation) and self.name == other.name

    def __repr__(self):
        return self.label

    def get_label(self):
        return self.label

    def add_input_name(self, name:str):
        self.inputs.append(name)
    
    def get_name(self)->str:
        return self.name

class ConstOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Const"):
        super().__init__(name, node, OperationType.CONST, label)
        self.value=torch.frombuffer(node.attribute[0].t.raw_data, dtype=torch.float32) # sus
        self.shape=list(self.value.shape)
        self.inputs=[]

    def get_label(self):
        printable = self.name + ":" 
        printable += "\nX" + str(self.shape)
        return printable

class MatMulOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="MatMul"):
        super().__init__(name, node, OperationType.UNKNOWN, label)

    def get_label(self):
        printable = self.name
        return printable

class TransposeOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Transpose"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class DivOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Div"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class ClipOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Clip"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class MulOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Mul"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class FloorOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Floor"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class AddOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Add"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable
        
class SubOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Sub"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class ReluOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Relu"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class ReshapeOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Reshape"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class TensorOP(Operation): # not op
    def __init__(self, name:str, tensor:torch.Tensor, label:str="Tensor"):
        super().__init__(name, None, OperationType.UNKNOWN, label)
        self.tensor=tensor

    def get_label(self):
        printable = self.name
        printable += "\nT" + str(list(self.tensor.shape))
        return printable

class ConvOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Conv"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node.attrib ute[2].name)

    def get_label(self):
        printable = self.name
        return printable

class MaxPoolOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="MaxPool"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class ModOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Mod"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class ShapeOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Shape"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable


class SliceOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Slice"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class ConcatOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Concat"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class SqueezeOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Squeeze"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class UnsqueezeOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Unsqueeze"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class SoftMaxOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="SoftMax"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class GatherOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Gather"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable

class GemmOP(Operation):
    def __init__(self, name:str, node:onnx.NodeProto, label:str="Gemm"):
        super().__init__(name, node, OperationType.UNKNOWN, label)
        # print(node)

    def get_label(self):
        printable = self.name
        return printable