import torch
from enum import Enum
import onnx

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
    def __init__(self, name: str, node: onnx.NodeProto, type: OperationType = OperationType.UNKNOWN, label: str = "OP"):
        self.name = name
        self.label = label
        self.type = type
        if node is not None:
            self.inputs = node.input
            self.outputs = node.output
        else:
            self.inputs = []
            self.outputs = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Operation) and self.name == other.name

    def __repr__(self):
        return self.label

    def get_label(self):
        return self.label

    def add_input_name(self, name: str):
        self.inputs.append(name)

    def get_name(self) -> str:
        return self.name

class ConstOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Const"):
        super().__init__(name, node, OperationType.CONST, label)
        self.tensor = torch.frombuffer(node.attribute[0].t.raw_data, dtype=torch.float32)
        self.inputs = []

    def get_label(self):
        return f"{self.label}:\nX{list(self.tensor.shape)}"

class MatMulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "MatMul"):
        super().__init__(name, node, OperationType.MATMUL, label)

    def get_label(self):
        return self.label

class TransposeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Transpose"):
        super().__init__(name, node, OperationType.TRANSPOSE, label)
        self.perm = [attr.ints for attr in node.attribute if attr.name == "perm"][0]

    def get_label(self):
        return f"{self.label}\nPerm: {self.perm}"

class DivOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Div"):
        super().__init__(name, node, OperationType.DIV, label)

    def get_label(self):
        return self.label

class ClipOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Clip"):
        super().__init__(name, node, OperationType.CLIP, label)

    def get_label(self):
        return self.label

class MulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Mul"):
        super().__init__(name, node, OperationType.MULT, label)

    def get_label(self):
        return self.label

class FloorOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Floor"):
        super().__init__(name, node, OperationType.FLOOR, label)

    def get_label(self):
        return self.label

class AddOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Add"):
        super().__init__(name, node, OperationType.ADD, label)

    def get_label(self):
        return self.label

class SubOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Sub"):
        super().__init__(name, node, OperationType.SUB, label)

    def get_label(self):
        return self.label

class ReluOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Relu"):
        super().__init__(name, node, OperationType.RELU, label)

    def get_label(self):
        return self.label

class ReshapeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Reshape"):
        super().__init__(name, node, OperationType.RESHAPE, label)
        self.allowzero = None
        if len(node.attribute):
            self.allowzero = [attr.i for attr in node.attribute if attr.name == "allowzero"]

    def get_label(self):
        return f"{self.label}\nallowzero: {self.allowzero}"

class TensorOP(Operation):
    def __init__(self, name: str, tensor: torch.Tensor, label: str = "Tensor"):
        super().__init__(name, None, OperationType.TENSOR, label)
        self.tensor = tensor

    def get_label(self):
        return f"{self.label}\nT{list(self.tensor.shape)}"

class ConvOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Conv"):
        super().__init__(name, node, OperationType.CONV, label)
        self.kernel_shape = [attr.ints for attr in node.attribute if attr.name == "kernel_shape"][0]
        self.strides = [attr.ints for attr in node.attribute if attr.name == "strides"][0]
        self.padding = [attr.ints for attr in node.attribute if attr.name == "pads"][0]
        self.group = [attr.ints for attr in node.attribute if attr.name == "group"][0]
        self.dilations = [attr.ints for attr in node.attribute if attr.name == "dilations"][0]

    def get_label(self):
        return f"{self.label}\nKernel Shape: {self.kernel_shape}\nStrides: {self.strides}\nPadding: {self.padding}"

class MaxPoolOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "MaxPool"):
        super().__init__(name, node, OperationType.MAXPOOL, label)
        self.kernel_shape = [attr.ints for attr in node.attribute if attr.name == "kernel_shape"][0]
        self.strides = [attr.ints for attr in node.attribute if attr.name == "strides"][0]
        self.padding = [attr.ints for attr in node.attribute if attr.name == "pads"][0]
        self.dilations = [attr.ints for attr in node.attribute if attr.name == "dilations"][0]
        self.ceil_mode = [attr.i for attr in node.attribute if attr.name == "ceil_mode"][0]

    def get_label(self):
        return f"{self.label}\nKernel Shape: {self.kernel_shape}\nStrides: {self.strides}\nPadding: {self.padding}"

class ModOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Mod"):
        super().__init__(name, node, OperationType.MOD, label)

    def get_label(self):
        return self.label

class ShapeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Shape"):
        super().__init__(name, node, OperationType.SHAPE, label)

    def get_label(self):
        return self.label

class SliceOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Slice"):
        super().__init__(name, node, OperationType.SLICE, label)

    def get_label(self):
        return self.label

class ConcatOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Concat"):
        super().__init__(name, node, OperationType.CONCAT, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\naxis: {self.axis}"

class SqueezeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Squeeze"):
        super().__init__(name, node, OperationType.SQUEEZE, label)

    def get_label(self):
        return self.label

class UnsqueezeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Unsqueeze"):
        super().__init__(name, node, OperationType.UNSQUEEZE, label)

    def get_label(self):
        return self.label

class SoftMaxOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "SoftMax"):
        super().__init__(name, node, OperationType.SOFTMAX, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\naxis: {self.axis}"

class GatherOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Gather"):
        super().__init__(name, node, OperationType.GATHER, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\naxis: {self.axis}"

class GemmOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Gemm"):
        super().__init__(name, node, OperationType.GEMM, label)
        self.alpha = [attr.i for attr in node.attribute if attr.name == "alpha"][0]
        self.beta = [attr.i for attr in node.attribute if attr.name == "beta"][0]
        self.transB = [attr.i for attr in node.attribute if attr.name == "transB"][0]

    def get_label(self):
        return f"{self.label}\nalpha: {self.alpha}\nbeta: {self.beta}\ntransB: {self.transB}"

############### custom operations ####################
class MacOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, label: str = "Gemm"):
        super().__init__(name, node, OperationType.GEMM, label)
        self.alpha = [attr.i for attr in node.attribute if attr.name == "alpha"][0]
        self.beta = [attr.i for attr in node.attribute if attr.name == "beta"][0]
        self.transB = [attr.i for attr in node.attribute if attr.name == "transB"][0]

    def get_label(self):
        return f"{self.label}\nalpha: {self.alpha}\nbeta: {self.beta}\ntransB: {self.transB}"
