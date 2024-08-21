import torch
from enum import Enum
import onnx
import networkx as nx
from nngraph.primitive import *
from graphviz import Digraph


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

class Operation(nx.DiGraph):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape=None, output_shape=None, type: OperationType = OperationType.UNKNOWN, label: str = "OP"):
        super().__init__()
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
        self.render()

    def render(self):
        dot = Digraph(self.name)
        dot.attr(label=self.label,
                style='dashed',
                color='black', 
                penwidth='2')

        for node in self.nodes():
            color = 'lightgrey'
            shape = 'box'
            style = 'filled'

            # if isinstance(node, ConstOP) or isinstance(node, TensorOP):
            #     color = 'lightblue'

            dot.node(node.get_name(), node.get_label(), color=color, shape=shape, style=style)

            for input_name in node.inputs:
                input_name=str(input_name).replace("::","/")
                known_inputs = []
                for pred in self.predecessors(node):
                    known_inputs+=pred.outputs
                if input_name not in known_inputs:
                    dot.edge(input_name, node.get_name())

        for edge in self.edges():
            dot.edge(edge[0].get_name(), edge[1].get_name())

        dot.render('./nngraph/outputs/primitive/' + self.name, format='png', cleanup=True) 

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
    
    def get_node(self, name:str)->Primitive:
        for node in self.nodes():
            if node.get_name() == name:
                return node
        return None
    
class ConstOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Const"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CONST, label)
        self.tensor = torch.frombuffer(node.attribute[0].t.raw_data, dtype=torch.float32)

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nTensor shape: {list(self.tensor.shape)}"

class MatMulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "MatMul"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MATMUL, label)
        self.__build_linear()
        self.render()

    def __build_linear(self):
        # blue
        self.add_node(Primitive("Weight" + self.name, label="Weight"))
        self.get_node("Weight" + self.name).inputs.append(self.inputs[1])
        
        self.add_node(Primitive("Input" + self.name, label="Input"))
        self.get_node("Input" + self.name).inputs.append(self.inputs[0])
        
        # green
        self.add_node(Primitive("Output" + self.name, label="Output"))
        self.get_node("Output" + self.name).outputs = self.outputs
        # self.outputs.append(self.get_node("Output" + self.name))

        # macs
        if (self.output_shape[1]*self.input_shape[1]>10000):
            print("operation too big")
            return

        for i in range(self.output_shape[1]):
            self.add_node(Primitive('Output_c' + str(i) + self.name, label="Output Channel" + str(i)))
            for j in range(self.input_shape[1]):
                mac=None
                mac_node_name="MAC" + self.name + str(i) + str(j)
                mac=MacPrim(mac_node_name, None, label="MAC" + str(i) + "," + str(j))
                self.add_node(mac) 
                self.add_edge(self.get_node('Input' + self.name), self.get_node(mac_node_name))
                self.add_edge(self.get_node('Weight' + self.name), self.get_node(mac_node_name))
                self.add_edge(self.get_node(mac_node_name), self.get_node('Output_c' + str(i) + self.name))
        
        for i in range(self.output_shape[1]):
            self.add_edge(self.get_node('Output_c' + str(i) + self.name), self.get_node('Output' + self.name))

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



    # def __build_conv(self, known_ops, node, conv):
    #     weights = self.get_input_tensor(conv)

    #     # blue
    #     self.add_node(Operation("Weight" + node.name, None, list(weights.tensor.shape), list(weights.tensor.shape), label="Weight"))
    #     self.get_node("Weight" + node.name).inputs.append(conv.inputs[1])
    #     self.add_node(Operation("Kernel" + node.name, None, list(weights.tensor.shape), [weights.tensor.shape[0]*weights.tensor.shape[1], 1], label="Kernel")) ##wrong
    #     self.add_edge(self.get_node("Weight" + node.name), self.get_node("Kernel" + node.name))
        
    #     input_shape, output_shape=self._get_in_out_shape(conv)
    #     self.add_node(Operation("Input" + node.name, None, input_shape, input_shape, label="Input"))
    #     self.get_node("Input" + node.name).inputs.append(conv.inputs[0])
        
    #     # grey
    #     self.add_node(Operation("Padding" + node.name, None, label="Padding"))
    #     self.add_node(Operation("Dilation" + node.name, None, label="Dilation"))
    #     self.add_edge(self.get_node("Kernel" + node.name), self.get_node("Dilation" + node.name))
    #     self.add_edge(self.get_node("Input" + node.name), self.get_node("Padding" + node.name))
        
    #     # green
    #     self.add_node(Operation("Output" + node.name, None, None, output_shape, label="Output"))
    #     self.get_node("Output" + node.name).outputs = conv.outputs
    #     self.outputs.append(self.get_node("Output" + node.name))
        
        
    #     # macs
    #     for i in range(conv.output_shape[1]):
    #         self.add_node(Operation('Output_c' + str(i) + node.name, None, label="Output Channel" + str(i)))
    #         for j in range(conv.input_shape[1]):
    #             mac=None
    #             mac_node_name="MAC" + node.name + str(i) + str(j)
    #             if len(conv.kernel_shape)==1:
    #                 mac=MacOP(mac_node_name, conv, label="MAC" + str(i) + "," + str(j))
    #             elif len(conv.kernel_shape)==2:
    #                 mac=Mac2dOP(mac_node_name, conv, label="MAC" + str(i) + "," + str(j))
    #             else:
    #                 print("invalid Convolution dims!")
        
    #             self.add_node(mac) 
    #             self.add_edge(self.get_node('Padding' + node.name), self.get_node(mac_node_name))
    #             self.add_edge(self.get_node('Dilation' + node.name), self.get_node(mac_node_name))
    #             self.add_edge(self.get_node(mac_node_name), self.get_node('Output_c' + str(i) + node.name))
        
        
        
    #     for i in range(conv.output_shape[1]):
    #         self.add_edge(self.get_node('Output_c' + str(i) + node.name), self.get_node('Output' + node.name))

    # def __build_maxpool(self, known_ops, node, conv):
    #     # blue
    #     self.add_node(Operation("Kernel" + node.name, None, label="Kernel"))
    #     self.add_node(Operation("Input" + node.name, None, label="Input"))
    #     self.get_node("Input" + node.name).inputs.append(conv.inputs[0])
        
    #     # grey
    #     self.add_node(Operation("Padding" + node.name, None, label="Padding"))
    #     self.add_node(Operation("Dilation" + node.name, None, label="Dilation"))
    #     self.add_edge(self.get_node("Kernel" + node.name), self.get_node("Dilation" + node.name))
    #     self.add_edge(self.get_node("Input" + node.name), self.get_node("Padding" + node.name))
        
    #     # green
    #     self.add_node(Operation("Output" + node.name, None, label="Output"))
    #     self.get_node("Output" + node.name).outputs = conv.outputs
    #     self.outputs.append(self.get_node("Output" + node.name))
        
    #     # macs
    #     for i in range(conv.output_shape[1]):
    #         self.add_node(Operation('Output_c' + str(i) + node.name,None, label="Output Channel" + str(i)))
    #         mac=None
    #         mac_node_name="MAC" + node.name + str(i)
    #         if len(conv.kernel_shape)==1:
    #             mac=MacOP(mac_node_name, conv, label="MAC" + str(i))
    #         elif len(conv.kernel_shape)==2:
    #             mac=Mac2dOP(mac_node_name, conv, label="MAC" + str(i))
    #         else:
    #             print("invalid Convolution dims!")
    
    #         self.add_node(mac) 
    #         self.add_edge(self.get_node('Padding' + node.name), self.get_node(mac_node_name))
    #         self.add_edge(self.get_node('Dilation' + node.name), self.get_node(mac_node_name))
    #         self.add_edge(self.get_node(mac_node_name), self.get_node('Output_c' + str(i) + node.name))
        
        
        
    #     for i in range(conv.input_shape[1]):
    #         self.add_edge(self.get_node('Output_c' + str(i) + node.name), self.get_node('Output' + node.name))


    # def get_input_tensor(self, op):
    #     weights_dict = {}
    #     for initializer in self.model_onnx.graph.initializer:
    #         tensor = torch.from_numpy(onnx.numpy_helper.to_array(initializer))
    #         weights_dict[initializer.name] = tensor
        
    #     all_inputs=[]
    #     weights=None
    #     for node in self.nodes:
    #         all_inputs += node.inputs
        
    #     for name, tensor in weights_dict.items():
    #         if name != op.inputs[1]:
    #             continue
    #         weights = TensorOP(name.replace("::", "/"), tensor)
    #     return weights
