import torch
from graphviz import Digraph
import torch.nn as nn
from enum import Enum
import networkx as nx
from nngraph.operation import *
import numpy as np
from onnx2torch import convert
import json

class LayerType(Enum):
    CONV1D=1
    CONV2D=2
    LINEAR=3
    MH_ATTENTION=4
    ACTIVATION=5
    UNKNOWN=0

class Layer(nx.DiGraph):
    def __init__(self, name:str, type:LayerType=LayerType.UNKNOWN, model_onnx=None, input_shape=[28,28], label:str="Layer"):
        super().__init__()
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
        self.outputs=[]
        self.model_onnx=model_onnx
        self.torch_model = convert(self.model_onnx)
        self.dummy_input = torch.randn(input_shape)

        self._build_operations()
        self._set_layer_inout()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['inputs']
        del state['outputs']
        del state['_node']
        del state['_adj']
        del state['_succ']
        del state['_pred']
        del state['succ']
        del state["_Layer__output_shape"]
        del state["_Layer__input_shape"]
        del state["nodes"]
        del state["edges"]
        super().__init__()

        return state

    def __setstate__(self, state):
        return
        self.__dict__.update(state)
        state['inputs']=[]
        state['outputs']=[]
        state['_node']=[]
        state['_adj']=[]
        state['_succ']=[]
        state['_pred']=[]
        state['succ']=[]
        state["_Layer__output_shape"]=[]
        state["_Layer__input_shape"]=[]
        state["nodes"]=[]
        state["edges"]=[]
        
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
                target_node = self.get_node(node.get_name())
                if len(self.edges) and len(node.outputs)>sum(1 for edge in self.edges() if edge[0] == target_node):
                    self.outputs.append(node)
                if len(self.edges) and len(node.inputs)>sum(1 for edge in self.edges() if edge[1] == target_node):
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
        with open("./nngraph/operation_config.json", 'r') as f:
            op_config = json.load(f)

        known_ops = {name: globals()[class_name] for name, class_name in op_config.items()}
        input_shape, output_shape = self._get_in_out_shape(node)

        if node.op_type not in known_ops.keys():
            print("unknown operation!")
            self.add_node(Operation(node.name, node, label=self.parse_onnx_op_name(node.name)))
        else:
            self.add_node(known_ops[node.op_type](node.name, node, input_shape, output_shape))
        
        self._add_tensors()
    
    def out_hook(self, module, input, output):
        self.__output_shape.append(list(output.shape))

    def get_operation_output_shape(self, model, layer_name, input_tensor):
        self.__output_shape = []

        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self.out_hook)

        with torch.no_grad():
            model(input_tensor)

        return self.__output_shape[0]

    def in_hook(self, module, input, output):
        for inp in input:
            if inp is not None:
                self.__input_shape.append(list(inp.shape))
            else:
                self.__input_shape.append(None)

    def get_operation_input_shape(self, model, layer_name, input_tensor):
        self.__input_shape = []

        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self.in_hook)

        with torch.no_grad():
            model(input_tensor)

        return self.__input_shape[0] if len(self.__input_shape)==1 else self.__input_shape

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

    def get_visual(self):
        dot = Digraph('cluster_' + self.name)
        dot.attr(label=self.label,
                style='dashed',
                color='black', 
                penwidth='2')

        for node in self.nodes():
            if isinstance(node, Layer):
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
