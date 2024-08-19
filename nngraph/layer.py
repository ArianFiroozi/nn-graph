import torch
from graphviz import Digraph
import pickle
import torch.nn as nn
from enum import Enum
import networkx as nx
from nngraph.operation import *

class LayerType(Enum):
    CONV1D=1
    CONV2D=2
    LINEAR=3
    MH_ATTENTION=4
    ACTIVATION=5
    UNKNOWN=0

class Layer(nx.DiGraph):
    def __init__(self, name:str, type:LayerType=LayerType.UNKNOWN, model_onnx=None, label:str="Layer", build_dummy_op=True, input_shape=[None]):
        super().__init__()
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
        self.outputs=[]
        self.input_shape=input_shape
        self.model_onnx=model_onnx

        # if build_dummy_op:
        self._build_operations()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Layer) and self.name == other.name

    def __repr__(self):
        return self.label

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
        name = '/'.join(name.split('/')[1:-1])
        return name if name!="" else name.split('/')[-1]

    def parse_onnx_op_name(self, name):
        return name.split('/')[-1]

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
            
    def _onnx_node_to_op(self, node):
        print(node.op_type)

        known_ops={"Constant":ConstOP, "MatMul":MatMulOP, "Transpose":TransposeOP}
        if node.op_type not in known_ops.keys():
            self.add_node(Operation(node.name, node, label=self.parse_onnx_op_name(node.name)))
        else:
            self.add_node(known_ops[node.op_type](node.name, node))

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

            if isinstance(node, ConstOP):
                color = 'lightblue'
            # elif isinstance(node, OutputOP):
            #     color = 'lightgreen'
            # if isinstance(node, PaddOP) or isinstance(node, DilationOP):
            #     shape = 'ellipse'

            dot.node(node.get_name(), node.get_label(), color=color, shape=shape, style=style)

            # for input_name in node.inputs:
            #     dot.edge(input_name, node.get_name())

        for edge in self.edges():
            dot.edge(edge[0].get_name(), edge[1].get_name())

        return dot
