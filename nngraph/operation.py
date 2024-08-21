import torch
from enum import Enum
import onnx
from nngraph.primitive import *

class OperationType(Enum):
    INPUT = "Input"
    MAC = "Multiply-Accumulate"
    ADD = "Addition"
    MULT = "Multiplication"
    OUTPUT = "Output"
    POOL = "Pooling"
    PADD = "Padding"
    DILATION = "Dilation"
    CONST = "Constant"
    PROJECT = "Projection"
    DOT_PRODUCT = "Dot Product"
    UNKNOWN = "Unknown"
    MATMUL = "Matrix Multiplication"
    TRANSPOSE = "Transpose"
    DIV = "Division"
    CLIP = "Clip"
    FLOOR = "Floor"
    SUB = "Subtraction"
    RELU = "ReLU"
    RESHAPE = "Reshape"
    TENSOR = "Tensor"
    CONV = "Convolution"
    MAXPOOL = "Max Pooling"
    MOD = "Modulus"
    SHAPE = "Shape"
    SLICE = "Slice"
    CONCAT = "Concatenation"
    SQUEEZE = "Squeeze"
    UNSQUEEZE = "Unsqueeze"
    SOFTMAX = "Softmax"
    GATHER = "Gather"
    GEMM = "General Matrix Multiply"

class Operation:
    def __init__(self, name: str, node: onnx.NodeProto, input_shape=None, output_shape=None, type: OperationType = OperationType.UNKNOWN, label: str = "OP"):
        self.name = name
        self.label = label
        self.type = type
        if node is not None:
            self.inputs = node.input
            self.outputs = node.output
        else:
            self.inputs = []
            self.outputs = []
        self.input_shape = None if input_shape is None else list(input_shape)
        self.output_shape = None if output_shape is None else list(output_shape)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Operation) and self.name == other.name

    def __repr__(self):
        return self.label

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}"

    def add_input_name(self, name: str):
        self.inputs.append(name)

    def get_name(self) -> str:
        return self.name

class ConstOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Const"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CONST, label)
        self.tensor = torch.frombuffer(node.attribute[0].t.raw_data, dtype=torch.float32)

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nTensor shape: {list(self.tensor.shape)}"

class MatMulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "MatMul"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MATMUL, label)

class TransposeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Transpose"):
        super().__init__(name, node, input_shape, output_shape, OperationType.TRANSPOSE, label)
        self.perm = [attr.ints for attr in node.attribute if attr.name == "perm"][0]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nPerm: {self.perm}"

class DivOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Div"):
        super().__init__(name, node, input_shape, output_shape, OperationType.DIV, label)

class ClipOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Clip"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CLIP, label)

class MulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Mul"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MULT, label)

class FloorOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Floor"):
        super().__init__(name, node, input_shape, output_shape, OperationType.FLOOR, label)

class AddOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Add"):
        super().__init__(name, node, input_shape, output_shape, OperationType.ADD, label)

class SubOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Sub"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SUB, label)

class ReluOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Relu"):
        super().__init__(name, node, input_shape, output_shape, OperationType.RELU, label)

class ReshapeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Reshape"):
        super().__init__(name, node, input_shape, output_shape, OperationType.RESHAPE, label)
        self.allowzero = None
        if len(node.attribute):
            self.allowzero = [attr.i for attr in node.attribute if attr.name == "allowzero"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nallowzero: {self.allowzero}"

class TensorOP(Operation):
    def __init__(self, name: str, tensor: torch.Tensor, label: str = "Tensor"):
        super().__init__(name, None, type=OperationType.TENSOR, label=label)
        self.tensor = tensor

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nTensor shape: {list(self.tensor.shape)}"

class ConvOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Conv"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CONV, label)
        self.kernel_shape = [attr.ints for attr in node.attribute if attr.name == "kernel_shape"][0]
        self.strides = [attr.ints for attr in node.attribute if attr.name == "strides"][0]
        self.padding = [attr.ints for attr in node.attribute if attr.name == "pads"][0]
        self.dilations = [attr.ints for attr in node.attribute if attr.name == "dilations"][0]
        self.group = [attr.ints for attr in node.attribute if attr.name == "group"][0]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nKernel Shape: {self.kernel_shape}\nStrides: {self.strides}\nPadding: {self.padding}"

class MaxPoolOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "MaxPool"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MAXPOOL, label)
        self.kernel_shape = [attr.ints for attr in node.attribute if attr.name == "kernel_shape"][0]
        self.strides = [attr.ints for attr in node.attribute if attr.name == "strides"][0]
        self.padding = [attr.ints for attr in node.attribute if attr.name == "pads"][0]
        self.dilations = [attr.ints for attr in node.attribute if attr.name == "dilations"][0]
        self.ceil_mode = [attr.i for attr in node.attribute if attr.name == "ceil_mode"][0]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nKernel Shape: {self.kernel_shape}\nStrides: {self.strides}\nPadding: {self.padding}"

class ModOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Mod"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MOD, label)

class ShapeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Shape"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SHAPE, label)

class SliceOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Slice"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SLICE, label)

class ConcatOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Concat"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CONCAT, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\naxis: {self.axis}"

class SqueezeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Squeeze"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SQUEEZE, label)

class UnsqueezeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Unsqueeze"):
        super().__init__(name, node, input_shape, output_shape, OperationType.UNSQUEEZE, label)

class SoftMaxOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "SoftMax"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SOFTMAX, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\naxis: {self.axis}"

class GatherOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Gather"):
        super().__init__(name, node, input_shape, output_shape, OperationType.GATHER, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\naxis: {self.axis}"

class GemmOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Gemm"):
        super().__init__(name, node, input_shape, output_shape, OperationType.GEMM, label)
        self.alpha = [attr.i for attr in node.attribute if attr.name == "alpha"][0]
        self.beta = [attr.i for attr in node.attribute if attr.name == "beta"][0]
        self.transB = [attr.i for attr in node.attribute if attr.name == "transB"][0]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nalpha: {self.alpha}\nbeta: {self.beta}\ntransB: {self.transB}"