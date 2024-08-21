import torch
from graphviz import Digraph
import torch.nn as nn
from enum import Enum
import networkx as nx
from nngraph.operation import *
import numpy as np
from onnx2torch import convert

class LayerType(Enum):
    CONV1D=1
    CONV2D=2
    LINEAR=3
    MH_ATTENTION=4
    ACTIVATION=5
    UNKNOWN=0

class Layer(nx.DiGraph):
    def __init__(self, name:str, type:LayerType=LayerType.UNKNOWN, model_onnx=None, label:str="Layer"):
        super().__init__()
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
        self.outputs=[]
        self.model_onnx=model_onnx
        self.torch_model = convert(self.model_onnx)
        self.dummy_input = torch.randn(28, 28)

        self._build_operations()
        self._set_layer_inout()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Layer) and self.name == other.name

    def __repr__(self):
        return self.label
    
    def _set_layer_inout(self):
            for node in self.nodes():
                if len(self.edges) and len(node.outputs)>np.count_nonzero(np.array(self.edges)[:,0]==self.get_node(node.get_name())):
                    self.outputs.append(node)
                if len(self.edges) and len(node.inputs)>np.count_nonzero(np.array(self.edges)[:,1]==self.get_node(node.get_name())):
                    self.inputs.append(node)
                if not len(self.edges) and len(self.nodes):
                    if len(node.inputs):
                        self.inputs.append(node)
                    if len(node.outputs):
                        self.outputs.append(node)
            return node

    def get_torch(self):
        print("layer: unknown layer type->"+self.name)
        return None

    def add_input(self, node):
        self.inputs.append(node)

    def add_output(self, node):
        self.outputs.append(node)

    def get_node(self, name:str)->Operation:
        for node in self.nodes():
            if node.get_name() == name:
                return node
        return None
    
    def get_name(self)->str:
        return self.name

    def parse_onnx_layer_name(self, name):
        if name.count("/")==1: #special case, handled bad
            name = name[1:]
        else:
            name = '/'.join(name.split('/')[1:-1])
            name = name if name!="" else name.split('/')[-1]
        return name

    def parse_onnx_op_name(self, name):
        return name.split('/')[-1]

    def _onnx_node_to_op(self, node):
        known_ops={"Constant":ConstOP, "MatMul":MatMulOP, "Transpose":TransposeOP, "Div":DivOP, "Clip":ClipOP,
                    "Mul":MulOP, "Floor":FloorOP, "Add":AddOP, "Sub":SubOP, "Relu":ReluOP, "Reshape":ReshapeOP,
                    "Conv":ConvOP, "MaxPool":MaxPoolOP, "Mod":ModOP, "Shape":ShapeOP,"Slice":SliceOP,"Concat":ConcatOP, 
                    "Squeeze":SqueezeOP,"Unsqueeze":UnsqueezeOP,"Softmax":SoftMaxOP,"Gather":GatherOP,"Gemm":GemmOP}
        
        input_shape, output_shape = self._get_in_out_shape(node)

        if node.op_type not in known_ops.keys():
            print("unknown operation!")
            self.add_node(Operation(node.name, node, label=self.parse_onnx_op_name(node.name)))
        else:
            self.add_node(known_ops[node.op_type](node.name, node, input_shape, output_shape))
        
        self._add_tensors()
    
    def get_operation_output_shape(self, model, layer_name, input_tensor):
        output_shape = []

        def hook(module, input, output):
            output_shape.append(output.shape)

        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(hook)

        with torch.no_grad():
            model(input_tensor)

        return output_shape[0]

    def get_operation_input_shape(self, model, layer_name, input_tensor):
        input_shape = []

        def hook(module, input, output):
            if len(input):
                input_shape.append(input[0].shape)
            else:
                input_shape.append(None)

        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(hook)

        with torch.no_grad():
            model(input_tensor)

        return input_shape[0]

    def _build_operations(self):
        for node in self.model_onnx.graph.node:
            name = self.parse_onnx_layer_name(node.name)
            if name==self.name:
                self._onnx_node_to_op(node)

        for first in self.nodes():
            for second in self.nodes():
                for output in first.outputs:
                    if output in second.inputs: 
                        self.add_edge(self.get_node(first.get_name()), self.get_node(second.get_name()))

    def _add_tensors(self):
        weights_dict = {}
        for initializer in self.model_onnx.graph.initializer:
            tensor = torch.from_numpy(onnx.numpy_helper.to_array(initializer))
            weights_dict[initializer.name] = tensor
        
        all_inputs=[]
        for node in self.nodes:
            all_inputs += node.inputs
        
        for name, tensor in weights_dict.items():
            if name not in all_inputs:
                continue
            tensor = TensorOP(name.replace("::", "/"), tensor)
            tensor.outputs=[tensor.name]
            self.add_node(tensor)

    def _get_in_out_shape(self, node, known_ops):
        if node.op_type in known_ops.keys():
            return self._get_in_out_shape(known_ops[node.op_type])
        return None, None

    def _get_in_out_shape(self, op):
        output_shape = self.get_operation_output_shape(self.torch_model, op.name[1:], self.dummy_input)
        input_shape = self.get_operation_input_shape(self.torch_model, op.name[1:], self.dummy_input)
        
        return input_shape, output_shape

    def get_output_shape(self):
        output = self.outputs[0] ##only one
        assert(isinstance(output, OutputOP))
        return output.shape

    def get_visual(self):
        dot = Digraph('cluster_' + self.name)
        dot.attr(label=self.label,
                style='dashed',
                color='black', 
                penwidth='2')

        for node in self.nodes():
            if isinstance(node, Layer): ## idk
                dot.subgraph(node.get_visual)
                continue

            color = 'lightgrey'
            shape = 'box'
            style = 'filled'

            if isinstance(node, ConstOP) or isinstance(node, TensorOP):
                color = 'lightblue'

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

        return dot
