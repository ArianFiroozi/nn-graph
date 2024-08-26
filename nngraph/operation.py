import torch
from enum import Enum
import onnx
import networkx as nx
import numpy as np
from nngraph.primitive import *
from graphviz import Digraph

class OperationType(Enum):
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
    UNKNOWN = "Unknown"

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

        self._build_primitives()
        self.render()

    def _build_primitives(self):
        pass

    def render(self):
        dot = Digraph(self.name)
        dot.attr(label=self.label,
                style='dashed',
                color='black', 
                penwidth='2')

        if (len(self.nodes)>1000):
            print("operation too big to render")
            dot.attr(label = self.label + " too big")
            dot.render('./nngraph/outputs/primitive/' + self.name, format='png', cleanup=True) 
            return

        for node in self.nodes():
            color = 'lightgrey'
            shape = 'box'
            style = 'filled'

            if isinstance(node, InputPrim):
                color = 'lightblue'
            if isinstance(node, OutputPrim):
                color = 'lightgreen'

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
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape=None, label: str = "Const"):
        self.tensor = torch.frombuffer(node.attribute[0].t.raw_data, dtype=torch.float32)
        super().__init__(name, node, input_shape, self.tensor.shape, OperationType.CONST, label)

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nTensor shape: {list(self.tensor.shape)}"

class DivOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Div"):
        super().__init__(name, node, input_shape, output_shape, OperationType.DIV, label)

    def _build_primitives(self):
        self.add_node(InputPrim("InputA"+self.name, self.input_shape[0]))
        self.add_node(InputPrim("InputB"+self.name, self.input_shape[0]))
        self.add_node(DivPrim("Div"+self.name))
        self.add_edge(self.get_node("InputA"+self.name), self.get_node("Div"+self.name))
        self.add_edge(self.get_node("InputB"+self.name), self.get_node("Div"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Div"+self.name), self.get_node("Output"+self.name))

class ClipOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Clip"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CLIP, label)
    
    def _build_primitives(self):
        input_num = len(self.input_shape) if isinstance(self.input_shape[0], list) else 1
        self.add_node(InputPrim("Input"+self.name, self.input_shape if input_num==1 else self.input_shape[0]))
        self.add_node(MaxPrim("Max"+self.name))
        self.add_node(MinPrim("Min"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Max"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Min"+self.name))

        if input_num>1:
            self.add_node(Input("InputMin"+self.name, self.input_shape[1], label="min"))
            self.add_edge(self.get_node("InputMin"+self.name), self.get_node("Min"+self.name))
        if input_num==3:
            self.add_node(Input("InputMax"+self.name, self.input_shape[1], label="max"))
            self.add_edge(self.get_node("InputMax"+self.name), self.get_node("Max"+self.name))
        
        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Min"+self.name), self.get_node("Output"+self.name))

class FloorOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Floor"):
        super().__init__(name, node, input_shape, output_shape, OperationType.FLOOR, label)
    
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape[0]))
        self.add_node(FloorPrim("Floor"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Floor"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Floor"+self.name), self.get_node("Output"+self.name))

class AddOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Add"):
        super().__init__(name, node, input_shape, output_shape, OperationType.ADD, label)

    def _build_primitives(self):
        self.add_node(InputPrim("InputA"+self.name, self.input_shape[0]))
        self.add_node(InputPrim("InputB"+self.name, self.input_shape[0]))
        self.add_node(AddPrim("Add"+self.name))
        self.add_edge(self.get_node("InputA"+self.name), self.get_node("Add"+self.name))
        self.add_edge(self.get_node("InputB"+self.name), self.get_node("Add"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Add"+self.name), self.get_node("Output"+self.name))

class SubOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Sub"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SUB, label)

    def _build_primitives(self):
        self.add_node(InputPrim("InputA"+self.name, self.input_shape[0]))
        self.add_node(InputPrim("InputB"+self.name, self.input_shape[0]))
        self.add_node(SubPrim("Sub"+self.name))
        self.add_edge(self.get_node("InputA"+self.name), self.get_node("Sub"+self.name))
        self.add_edge(self.get_node("InputB"+self.name), self.get_node("Sub"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Sub"+self.name), self.get_node("Output"+self.name))

class ReluOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Relu"):
        super().__init__(name, node, input_shape, output_shape, OperationType.RELU, label)

    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape[0]))
        self.add_node(Primitive("Zero"+self.name, label="Zero"))
        self.add_node(MaxPrim("Max"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Max"+self.name))
        self.add_edge(self.get_node("Zero"+self.name), self.get_node("Max"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Max"+self.name), self.get_node("Output"+self.name))

class ReshapeOP(Operation): 
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Reshape"):
        super().__init__(name, node, input_shape, output_shape, OperationType.RESHAPE, label)
        self.allowzero = None
        if len(node.attribute):
            self.allowzero = [attr.i for attr in node.attribute if attr.name == "allowzero"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nallowzero: {self.allowzero}"
    
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape[0]))
        self.add_node(InputPrim("Shape"+self.name, self.input_shape[1]))
        self.add_node(ReshapePrim("Reshape"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Reshape"+self.name))
        self.add_edge(self.get_node("Shape"+self.name), self.get_node("Reshape"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Reshape"+self.name), self.get_node("Output"+self.name))

class TensorOP(Operation):
    def __init__(self, name: str, tensor: torch.Tensor, label: str = "Tensor"):
        super().__init__(name, None, type=OperationType.TENSOR, label=label)
        self.tensor = tensor

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nTensor shape: {list(self.tensor.shape)}"

class ModOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Mod"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MOD, label)

    def _build_primitives(self):
        self.add_node(InputPrim("InputA"+self.name, self.input_shape[0]))
        self.add_node(InputPrim("InputB"+self.name, self.input_shape[0]))
        self.add_node(ModPrim("Mod"+self.name))
        self.add_edge(self.get_node("InputA"+self.name), self.get_node("Mod"+self.name))
        self.add_edge(self.get_node("InputB"+self.name), self.get_node("Mod"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Mod"+self.name), self.get_node("Output"+self.name))

class ShapeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Shape"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SHAPE, label)
    
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape))
        self.add_node(ShapePrim("Shape"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Shape"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Shape"+self.name), self.get_node("Output"+self.name))

class SliceOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Slice"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SLICE, label)
        
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape[0],label="Data"))
        self.add_node(InputPrim("InputStart"+self.name, self.input_shape[1],label="Start"))
        self.add_node(InputPrim("InputEnd"+self.name, self.input_shape[2],label="End"))
        self.add_node(SlicePrim("Slice"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Slice"+self.name))
        self.add_edge(self.get_node("InputStart"+self.name), self.get_node("Slice"+self.name))
        self.add_edge(self.get_node("InputEnd"+self.name), self.get_node("Slice"+self.name))

        if len(self.input_shape)>=4:
            self.add_node(InputPrim("InputAxes"+self.name, self.input_shape[3],label="Axes"))
            self.add_edge(self.get_node("InputAxes"+self.name), self.get_node("Slice"+self.name))
        if len(self.input_shape)==5:
            self.add_node(InputPrim("InputSteps"+self.name, self.input_shape[4],label="Steps"))
            self.add_edge(self.get_node("InputSteps"+self.name), self.get_node("Slice"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Slice"+self.name), self.get_node("Output"+self.name))

class TransposeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Transpose"):
        super().__init__(name, node, input_shape, output_shape, OperationType.TRANSPOSE, label)
        self.perm = [attr.ints for attr in node.attribute if attr.name == "perm"][0]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nPerm: {self.perm}"
    
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape))
        self.add_node(TransposePrim("Transpose"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Transpose"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Transpose"+self.name), self.get_node("Output"+self.name))

class ConcatOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Concat"):
        super().__init__(name, node, input_shape, output_shape, OperationType.CONCAT, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\naxis: {self.axis}"
    
    def _build_primitives(self):
        input_num = len(self.input_shape) if isinstance(self.input_shape[0], list) else 1
        self.add_node(ConcatPrim("Concat"+self.name))

        if input_num==1:
            self.add_node(InputPrim("Input"+self.name, self.input_shape))
            self.add_edge(self.get_node("Input"+self.name), self.get_node("Concat"+self.name))
        else:
            for i in range(input_num):
                self.add_node(InputPrim("Input"+str(i)+self.name, self.input_shape[i]))
                self.add_edge(self.get_node("Input"+str(i)+self.name), self.get_node("Concat"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Concat"+self.name), self.get_node("Output"+self.name))

class SqueezeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Squeeze"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SQUEEZE, label)
    
    def _build_primitives(self):
        input_num = len(self.input_shape) if isinstance(self.input_shape[0], list) else 1

        self.add_node(InputPrim("Input"+self.name, self.input_shape[0] if input_num>1 else self.input_shape))
        self.add_node(SqueezePrim("Squeeze"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Squeeze"+self.name))

        if input_num==2:
            self.add_node(InputPrim("Axes"+self.name, self.input_shape[1]))
            self.add_edge(self.get_node("Axes"+self.name), self.get_node("Squeeze"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Squeeze"+self.name), self.get_node("Output"+self.name))

class UnsqueezeOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Unsqueeze"):
        super().__init__(name, node, input_shape, output_shape, OperationType.UNSQUEEZE, label)
            
    def _build_primitives(self):
        input_num = len(self.input_shape) if isinstance(self.input_shape[0], list) else 1

        self.add_node(InputPrim("Input"+self.name, self.input_shape[0] if input_num>1 else self.input_shape))
        self.add_node(UnsqueezePrim("Unsqueeze"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Unsqueeze"+self.name))

        if input_num==2:
            self.add_node(InputPrim("Axes"+self.name, self.input_shape[1]))
            self.add_edge(self.get_node("Axes"+self.name), self.get_node("Unsqueeze"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Unsqueeze"+self.name), self.get_node("Output"+self.name))

class SoftMaxOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "SoftMax"):
        super().__init__(name, node, input_shape, output_shape, OperationType.SOFTMAX, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\naxis: {self.axis}"
    
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape))
        self.add_node(SoftMaxPrim("SoftMax"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("SoftMax"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("SoftMax"+self.name), self.get_node("Output"+self.name))

class MulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Mul"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MULT, label)

    def _build_primitives(self):
        self.add_node(InputPrim("InputA"+self.name, self.input_shape[0]))
        self.add_node(InputPrim("InputB"+self.name, self.input_shape[0]))
        self.add_node(MulPrim("Mul"+self.name))
        self.add_edge(self.get_node("InputA"+self.name), self.get_node("Mul"+self.name))
        self.add_edge(self.get_node("InputB"+self.name), self.get_node("Mul"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Mul"+self.name), self.get_node("Output"+self.name))

class GatherOP(Operation):  
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Gather"):
        super().__init__(name, node, input_shape, output_shape, OperationType.GATHER, label)
        self.axis = [attr.i for attr in node.attribute if attr.name == "axis"]

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\naxis: {self.axis}"
    
    def _build_primitives(self):
        self.add_node(InputPrim("Input"+self.name, self.input_shape[0]))
        self.add_node(GatherPrim("Gather"+self.name))
        self.add_edge(self.get_node("Input"+self.name), self.get_node("Gather"+self.name))
        self.add_node(InputPrim("Indices"+self.name, self.input_shape[1], label="Indices"))
        self.add_edge(self.get_node("Indices"+self.name), self.get_node("Gather"+self.name))

        self.add_node(OutputPrim("Output"+self.name, self.output_shape))
        self.add_edge(self.get_node("Gather"+self.name), self.get_node("Output"+self.name))

############has multiple primitives##############

class MaxPoolOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "MaxPool"):
        self.kernel_shape = [attr.ints for attr in node.attribute if attr.name == "kernel_shape"][0]
        self.strides = [attr.ints for attr in node.attribute if attr.name == "strides"][0]
        self.padding = [attr.ints for attr in node.attribute if attr.name == "pads"][0]
        self.dilations = [attr.ints for attr in node.attribute if attr.name == "dilations"][0]
        self.ceil_mode = [attr.i for attr in node.attribute if attr.name == "ceil_mode"][0]
        super().__init__(name, node, input_shape, output_shape, OperationType.MAXPOOL, label)

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nKernel Shape: {self.kernel_shape}\nStrides: {self.strides}\nPadding: {self.padding}"
    
    def _build_primitives(self):
        # blue
        kernel_name = "Kernel" + self.name
        kernel_node = InputPrim(kernel_name, self.kernel_shape, label="Kernel")
        self.add_node(kernel_node)

        input_name = "Input" + self.name
        input_node = InputPrim(input_name, self.input_shape, label="Input")
        self.add_node(input_node)
        input_node.inputs.append(self.inputs[0])

        # grey
        padding_name = "Padding" + self.name
        padding_node = PaddingPrim(padding_name, label="Padding")
        self.add_node(padding_node)

        dilation_name = "Dilation" + self.name
        dilation_node = DilationPrim(dilation_name, label="Dilation")
        self.add_node(dilation_node)

        self.add_edge(kernel_node, dilation_node)
        self.add_edge(input_node, padding_node)

        # green
        output_name = "Output" + self.name
        output_node = OutputPrim(output_name, self.output_shape, label="Output")
        self.add_node(output_node)
        output_node.outputs = self.outputs

        # macs
        for i in range(self.output_shape[1]):
            output_channel_name = 'Output_c' + str(i) + self.name
            output_channel_node = OutputPrim(output_channel_name, None, label="Output Channel" + str(i))
            self.add_node(output_channel_node)

            mac_node_name = "Pool" + self.name + str(i)
            if len(self.kernel_shape) == 1:
                mac_node = PoolPrim(mac_node_name, self, label="Pool" + str(i))
            elif len(self.kernel_shape) == 2:
                mac_node = Pool2dPrim(mac_node_name, self, label="Pool" + str(i))
            else:
                print("invalid MaxPool dims!")
                continue

            self.add_node(mac_node)
            self.add_edge(padding_node, mac_node)
            self.add_edge(dilation_node, mac_node)
            self.add_edge(mac_node, output_channel_node)
            self.add_edge(output_channel_node, output_node)

class ConvOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Conv"):
        self.kernel_shape = [attr.ints for attr in node.attribute if attr.name == "kernel_shape"][0]
        self.strides = [attr.ints for attr in node.attribute if attr.name == "strides"][0]
        self.padding = [attr.ints for attr in node.attribute if attr.name == "pads"][0]
        self.dilations = [attr.ints for attr in node.attribute if attr.name == "dilations"][0]
        self.group = [attr.ints for attr in node.attribute if attr.name == "group"][0]
        super().__init__(name, node, input_shape, output_shape, OperationType.CONV, label)

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nKernel Shape: {self.kernel_shape}\nStrides: {self.strides}\nPadding: {self.padding}"

    def _build_primitives(self):
        weights = []
        weights.append(self.output_shape[1])
        weights.append(self.input_shape[1])
        weights += list(self.kernel_shape)

        # blue
        weight_name = "Weight" + self.name
        weight_node = InputPrim(weight_name, list(weights), label="Weight")
        self.add_node(weight_node)
        weight_node.inputs.append(self.inputs[1])

        kernel_name = "Kernel" + self.name
        kernel_node = InputPrim(kernel_name, [weights[0] * weights[1], weights[2]], label="Kernel")  # Corrected
        self.add_node(kernel_node)
        self.add_edge(weight_node, kernel_node)

        input_name = "Input" + self.name
        input_node = InputPrim(input_name, self.input_shape, label="Input")
        self.add_node(input_node)
        input_node.inputs.append(self.inputs[0])

        # grey
        padding_name = "Padding" + self.name
        padding_node = PaddingPrim(padding_name, label="Padding")
        self.add_node(padding_node)

        dilation_name = "Dilation" + self.name
        dilation_node = DilationPrim(dilation_name, label="Dilation")
        self.add_node(dilation_node)

        self.add_edge(kernel_node, dilation_node)
        self.add_edge(input_node, padding_node)

        # green
        output_name = "Output" + self.name
        output_node = OutputPrim(output_name, self.output_shape, label="Output")
        self.add_node(output_node)
        output_node.outputs = self.outputs

        # macs
        for batch in range(self.output_shape[0]):
            batch_node = OutputPrim('batch' + str(batch) + self.name, None, label="batch" + str(batch))
            for i in range(self.output_shape[1]):
                output_channel_node = OutputPrim('Output_c' + str(i) + str(batch) + self.name, None, label="Output Channel" + str(i))
                add_i_node = AddPrim("Add" + str(i) + str(batch) + self.name, label="Add " + str(i))
                self.add_node(output_channel_node)

                for j in range(self.input_shape[1]):
                    for k in range(self.output_shape[1]):
                        mac_node_name = "MAC" + self.name + str(batch) + str(i) + str(j) + str(k)
                        if len(self.kernel_shape) == 1:
                            mac_node = MacPrim(mac_node_name, self, label="MAC" + str(i) + "," + str(j))
                            weight_indices = [(i, j, k) for k in range(self.kernel_shape[0])]
                        elif len(self.kernel_shape) == 2:
                            mac_node = Mac2dPrim(mac_node_name, self, label="MAC" + str(i) + "," + str(j))
                            weight_indices = [(i, j, ki, kj) for ki in range(self.kernel_shape[0]) for kj in range(self.kernel_shape[1])]
                        else:
                            print("invalid Convolution dims!")
                            continue
                        mac_node.weight_indices = weight_indices
                        mac_node.input_channel=j
                        mac_node.output_index=k
                        mac_node.batch=batch

                        self.add_node(mac_node)
                        self.add_edge(padding_node, mac_node)
                        self.add_edge(dilation_node, mac_node)
                        self.add_edge(mac_node, add_i_node)
                    
                self.add_edge(add_i_node, output_channel_node)
                self.add_edge(output_channel_node, batch_node)
            self.add_edge(batch_node, output_node)

class MatMulOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "MatMul"):
        super().__init__(name, node, input_shape, output_shape, OperationType.MATMUL, label)
    
    def _build_primitives(self):
        # blue
        weight=InputPrim("InputB" + self.name, self.input_shape[1], label="Input B")
        self.add_node(weight)
        self.get_node("InputB" + self.name).inputs.append(self.inputs[1])
        
        input=InputPrim("InputA" + self.name, self.input_shape[0], label="Input A")
        self.add_node(input)
        self.get_node("InputA" + self.name).inputs.append(self.inputs[0])
        
        # green
        output=OutputPrim("Output" + self.name, self.output_shape, label="Output")
        self.add_node(output)
        self.get_node("Output" + self.name).outputs = self.outputs

        # macs
        for i in range(np.prod(np.array(self.output_shape))):
            mac=None
            mac_node_name="Mac" + self.name + str(i)
            mac=MacPrim(mac_node_name, None, label="Mac")
            self.add_node(mac) 
            self.add_edge(input, mac)
            self.add_edge(weight, mac)
            self.add_edge(mac, output)

class GemmOP(Operation):
    def __init__(self, name: str, node: onnx.NodeProto, input_shape, output_shape, label: str = "Gemm"):
        self.alpha = [attr.i for attr in node.attribute if attr.name == "alpha"][0]
        self.beta = [attr.i for attr in node.attribute if attr.name == "beta"][0]
        self.transA = [attr.i for attr in node.attribute if attr.name == "transA"]
        self.transB = [attr.i for attr in node.attribute if attr.name == "transB"]
        self.transA = self.transA[0] if len(self.transA) else None
        self.transB = self.transB[0] if len(self.transB) else None
        super().__init__(name, node, input_shape, output_shape, OperationType.GEMM, label)

    def get_label(self):
        return f"{self.label}\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\nalpha: {self.alpha}\nbeta: {self.beta}\ntransB: {self.transB}\ntransA: {self.transA}"

    def _build_primitives(self):
        # blue
        weight_node = InputPrim("InputA" + self.name, None, label="Input A")
        self.add_node(weight_node)
        weight_node.inputs.append(self.inputs[1])

        input_node = InputPrim("InputB" + self.name, self.input_shape, label="Input B")
        self.add_node(input_node)
        input_node.inputs.append(self.inputs[0])

        bias_name = "Bias" + self.name
        bias_node = InputPrim(bias_name, None, label="Bias")
        self.add_node(bias_node)
        bias_node.inputs.append(self.inputs[2])

        # green
        output_name = "Output" + self.name
        output_node = OutputPrim(output_name, self.output_shape, label="Output")
        self.add_node(output_node)

        matmul_out_name = "MatMulOut" + self.name
        matmul_out_node = OutputPrim(matmul_out_name, None, label="MatMul Output")
        self.add_node(matmul_out_node)
        output_node.outputs = self.outputs

        if self.transA is not None: 
            transpose_a_name = "TransposeA" + self.name
            transpose_a_node = TransposePrim(transpose_a_name, label="Transpose")
            self.add_node(transpose_a_node)
            self.add_edge(input_node, transpose_a_node)
            input_node = transpose_a_node

        if self.transB is not None: 
            transpose_b_name = "TransposeB" + self.name
            transpose_b_node = TransposePrim(transpose_b_name, label="Transpose")
            self.add_node(transpose_b_node)
            self.add_edge(weight_node, transpose_b_node)
            weight_node = transpose_b_node

        # macs
        for i in range(np.prod(np.array(self.output_shape))):
            mac=None
            mac_node_name="Mac" + self.name + str(i)
            mac=MacPrim(mac_node_name, None, label="Mac")
            self.add_node(mac) 
            self.add_edge(input_node, mac)
            self.add_edge(weight_node, mac)
            self.add_edge(mac, output_node)

        mult_a_name = "MultA" + self.name
        mult_a_node = MulPrim(mult_a_name)
        self.add_node(mult_a_node)
        self.add_edge(matmul_out_node, mult_a_node)

        mult_b_name = "MultB" + self.name
        mult_b_node = MulPrim(mult_b_name)
        self.add_node(mult_b_node)
        self.add_edge(bias_node, mult_b_node)

        add_name = "ADD" + self.name
        add_node = AddPrim(add_name, label="ADD")
        self.add_node(add_node)
        self.add_edge(mult_a_node, add_node)
        self.add_edge(mult_b_node, add_node)
        self.add_edge(add_node, output_node)
