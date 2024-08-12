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
    UNKNOWN=0

class Layer(nx.DiGraph):
    def __init__(self, name:str, type:LayerType, label:str="OP", build_dummy_op=True):
        super().__init__()
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
        self.outputs=[]
        if build_dummy_op:
            self._build_operations()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Layer) and self.name == other.name

    def __repr__(self):
        return self.label

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

    def _build_operations(self):
        self.add_node(InputOP('Input'+self.name,[None]))
        self.add_input(self.get_node('Input'+self.name))
        self.add_node(Operation('OP'+self.name, label=self.name))
        self.add_node(OutputOP('Output'+self.name,[None]))
        self.add_output(self.get_node('Output'+self.name))

        self.add_edge(self.get_node('Input'+self.name), self.get_node('OP'+self.name))
        self.add_edge(self.get_node('OP'+self.name), self.get_node('Output'+self.name))

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

            if isinstance(node, InputOP) or isinstance(node, ConstOP):
                color = 'lightblue'
            elif isinstance(node, OutputOP):
                color = 'lightgreen'
            if isinstance(node, PaddOP) or isinstance(node, DilationOP):
                shape = 'ellipse'

            dot.node(node.get_name(), node.get_label(), color=color, shape=shape, style=style)

            for input_name in node.inputs:
                dot.edge(input_name, node.get_name())

        for edge in self.edges():
            dot.edge(edge[0].get_name(), edge[1].get_name())

        return dot

class Conv1dLayer(Layer):
    def __init__(self, name:str, in_channels, out_channels, kernel_size, input_length,
                stride=[1], padding=[1], dilation=[1], bias=None, weight=None, sparsity=0.0, label="Conv1d"):
        super().__init__(name, LayerType.CONV1D, label+": "+name, False)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size = kernel_size[0]
        self.input_length = input_length
        self.stride = stride[0]
        self.padding = padding[0]
        self.dilation = dilation[0]
        self.sparsity = sparsity
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias is None else bias
        
        self._build_operations()

    def get_torch(self):
        return nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )

    def _build_operations(self):
        # blue
        self.add_node(ConstOP("Weight" + self.name, [self.out_channels, self.kernel_size], "Weight"))
        self.add_node(ConstOP("Kernel" + self.name, [self.kernel_size], "Kernel"))
        self.add_edge(self.get_node('Weight' + self.name), self.get_node('Kernel' + self.name))
        self.add_node(InputOP("Input" + self.name, [self.in_channels, self.input_length]))
        self.add_input(self.get_node('Input'+self.name))

        #grey
        self.add_node(PaddOP("Padding" + self.name, self.padding))
        self.add_node(DilationOP("Dilation" + self.name, self.dilation))
        self.add_edge(self.get_node("Kernel" + self.name), self.get_node("Dilation" + self.name))
        self.add_edge(self.get_node("Input" + self.name), self.get_node("Padding" + self.name))

        #green
        self.add_node(OutputOP("Output" + self.name, [self.out_channels, self._calc_out_size()]))
        self.add_output(self.get_node('Output' + self.name))

        #macs
        for i in range(self.out_channels):
            self.add_node(OutputOP('Output_c' + str(i) + self.name, [self._calc_out_size()], "Output Channel" + str(i)))
            for j in range(self._calc_out_size()):
                mac_node_name = f'Mac{self.name}_{i}_{j}'
                input_start = j * self.stride - self.padding
                weight_start = 0 
                weight_end = self.kernel_size - 1
                
                mac = MacOP(mac_node_name, [input_start, input_start+self.kernel_size],
                                [weight_start, weight_end], "MAC" + str(i) + "," + str(j))
                if self.weight is not None:
                    mac.weight = self.weight[weight_start:weight_end]

                self.add_node(mac)                
                self.add_edge(self.get_node('Padding' + self.name), self.get_node(mac_node_name))
                self.add_edge(self.get_node('Dilation' + self.name), self.get_node(mac_node_name))
                self.add_edge(self.get_node(mac_node_name), self.get_node('Output_c' + str(i) + self.name))


        
        for i in range(self.out_channels):
            self.add_edge(self.get_node('Output_c' + str(i) + self.name), self.get_node('Output' + self.name))

    def _calc_out_size(self):
        return (self.input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

class Conv2dLayer(Layer):
    def __init__(self, name: str, in_channels, out_channels, kernel_size, input_height, input_width,
                 stride=[1, 1], padding=[1, 1], dilation=[1, 1], bias=None, weight=None, sparsity=0.0, label="Conv2d"):
        super().__init__(name, LayerType.CONV2D, label + ": " + name, False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        self.input_height = input_height
        self.input_width = input_width
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sparsity = sparsity
        self.weight=weight
        
        self._build_operations()

    def get_torch(self):
        return nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

    def _build_operations(self):
        # blue
        self.add_node(ConstOP("Weight" + self.name, [self.out_channels, self.in_channels] + list(self.kernel_size), "Weight"))
        self.add_node(ConstOP("Kernel" + self.name, list(self.kernel_size), "Kernel"))
        self.add_edge(self.get_node("Weight" + self.name), self.get_node("Kernel" + self.name))

        self.add_node(InputOP("Input" + self.name, [self.in_channels, self.input_height, self.input_width]))
        self.add_input(self.get_node("Input" + self.name))

        # grey
        self.add_node(PaddOP("Padding" + self.name, self.padding))
        self.add_node(DilationOP("Dilation" + self.name, self.dilation))
        self.add_edge(self.get_node("Kernel" + self.name), self.get_node("Dilation" + self.name))
        self.add_edge(self.get_node("Input" + self.name), self.get_node("Padding" + self.name))

        # green
        self.add_node(OutputOP("Output" + self.name, [self.out_channels, self._calc_out_height(), self._calc_out_width()]))
        self.add_output(self.get_node("Output" + self.name))

        # macs
        for i in range(self.out_channels):
            self.add_node(OutputOP('Output_c' + str(i) + self.name, [self._calc_out_height(), self._calc_out_width()], "Output Channel" + str(i)))
            for j in range(self._calc_out_height()):
                for k in range(self._calc_out_width()):
                    mac_node_name = f'Mac{self.name}_{i}_{j}_{k}'
                    input_start_h = j * self.stride[0] - self.padding[0]
                    input_start_w = k * self.stride[1] - self.padding[1]
                    weight_start_h = 0
                    weight_start_w = 0
                    weight_end_h = self.kernel_size[0] - 1
                    weight_end_w = self.kernel_size[1] - 1

                    mac = MacOP(mac_node_name, 
                                        [[input_start_h, input_start_h + self.kernel_size[0]], 
                                        [input_start_w, input_start_w + self.kernel_size[1]]],
                                        [[weight_start_h, weight_end_h], [weight_start_w, weight_end_w]], 
                                        "MAC" + str(i) + "," + str(j) + "," + str(k))
                    if isinstance(self.weight, torch.Tensor):
                        mac.weight = self.weight
                        print(self.weight.shape)

                    self.add_node(mac)                
                    self.add_edge(self.get_node('Padding' + self.name), self.get_node(mac_node_name))
                    self.add_edge(self.get_node('Dilation' + self.name), self.get_node(mac_node_name))
                    self.add_edge(self.get_node(mac_node_name), self.get_node('Output_c' + str(i) + self.name))

        for i in range(self.out_channels):
            self.add_edge(self.get_node('Output_c' + str(i) + self.name), self.get_node('Output' + self.name))


    def _calc_out_height(self):
        return (self.input_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

    def _calc_out_width(self):
        return (self.input_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

    class LinearLayer(Layer):
        def __init__(self, name, in_features, out_features, label="Linear"):
            super().__init__(name, LayerType.LINEAR,  label+": "+name, False)
            self.in_features=in_features
            self.out_features=out_features

            self._build_operations()

        def get_torch(self):
            return nn.Linear(   
                    in_features=int(self.in_features),
                    out_features=int(self.out_features)
                )

        def _build_operations(self):
            # blue
            self.add_node(ConstOP("Weight" + self.name, [self.out_features, self.in_features], "Weight"))
            self.add_node(InputOP("Input" + self.name, [self.in_features]))
            self.add_input(self.get_node('Input' + self.name))

            #green
            self.add_node(OutputOP("Output" + self.name, [self.out_features]))
            self.add_output(self.get_node('Output' + self.name))

            #macs
            for i in range(self.out_features):
                
                mac_node_name = f'Mac{self.name}_{i}'
                input_start = 0
                weight_start = 0 

                self.add_node(MacOP(mac_node_name, [input_start, input_start+self.in_features],
                                [i], "MAC" + str(i)))
                self.add_edge(self.get_node('Input' + self.name), self.get_node(mac_node_name))
                self.add_edge(self.get_node('Weight' + self.name), self.get_node(mac_node_name))
                self.add_edge(self.get_node(mac_node_name), self.get_node('Output' + self.name))

class LinearLayer(Layer):
    def __init__(self, name, in_features, out_features, label="Linear"):
        super().__init__(name, LayerType.LINEAR,  label+": "+name, False)
        self.in_features=in_features
        self.out_features=out_features

        self._build_operations()

    def get_torch(self):
        return nn.Linear(   
                in_features=int(self.in_features),
                out_features=int(self.out_features)
            )

    def _build_operations(self):
        # blue
        self.add_node(ConstOP("Weight" + self.name, [self.out_features, self.in_features], "Weight"))
        self.add_node(InputOP("Input" + self.name, [self.in_features]))
        self.add_input(self.get_node('Input'+self.name))

        #green
        self.add_node(OutputOP("Output" + self.name, [self.out_features]))
        self.add_output(self.get_node('Output'+self.name))

        #macs
        for i in range(self.out_features):
            
            mac_node_name = f'Mac{self.name}_{i}'
            input_start = 0
            weight_start = 0 

            self.add_node(MacOP(mac_node_name, [input_start, input_start+self.in_features],
                            [i], "MAC" + str(i)))
            self.add_edge(self.get_node('Input' + self.name), self.get_node(mac_node_name))
            self.add_edge(self.get_node('Weight' + self.name), self.get_node(mac_node_name))
            self.add_edge(self.get_node(mac_node_name), self.get_node('Output' + self.name))

class MHAttentionLayer(Layer):
    def __init__(self, name, input_shape, embed_dim, num_heads, label="MultiHeadAttention"):
        super().__init__(name, LayerType.MH_ATTENTION,  label+": "+name, False)
        self.input_shape=input_shape
        self.num_heads=num_heads
        self.embed_dim=embed_dim

        self._build_operations()

    def get_torch(self):
        return nn.MultiheadAttention(self.embed_dim, 
                                    self.num_heads)

    def _build_operations(self):
        # blue
        self.add_node(InputOP("Input" + self.name, [self.input_shape]))
        self.add_input(self.get_node('Input' + self.name))
        # self.add_node(ProjectOP("Project" + self.name, [self.input_shape]))
        # self.add_edge(self.get_node("Input" + self.name), self.get_node("Project" + self.name))

        self.add_node(InputOP("Q" + self.name, [self.embed_dim], "Q"))
        self.add_node(InputOP("K" + self.name, [self.embed_dim], "K"))
        self.add_node(InputOP("V" + self.name, [self.embed_dim], "V"))
        self.add_edge(self.get_node("Input" + self.name), self.get_node("Q" + self.name))
        self.add_edge(self.get_node("Input" + self.name), self.get_node("K" + self.name))
        self.add_edge(self.get_node("Input" + self.name), self.get_node("V" + self.name))

        for i in range(self.num_heads):
            self.add_node(InputOP("Q" + str(i) + self.name, [self.embed_dim // self.num_heads], "Q" + str(i)))
            self.add_node(InputOP("K" + str(i) + self.name, [self.embed_dim // self.num_heads], "K" + str(i)))
            self.add_node(InputOP("V" + str(i) + self.name, [self.embed_dim // self.num_heads], "V" + str(i)))
            self.add_edge(self.get_node("Q" + self.name), self.get_node("Q" + str(i) + self.name))
            self.add_edge(self.get_node("K" + self.name), self.get_node("K" + str(i) + self.name))
            self.add_edge(self.get_node("V" + self.name), self.get_node("V" + str(i) + self.name))

            self.add_node(OutputOP("HeadOutput" + str(i) + self.name, [self.input_shape[1] // self.num_heads], "Head Out" + str(i)))

        for i in range(self.num_heads):
            self.add_node(DotProduct("DOTQK" + str(i) + self.name, [None], [None], "Dot Q K " + str(i)))  # TODO
            self.add_node(Operation("SoftMax" + str(i) + self.name, label="SoftMax"))
            self.add_node(DotProduct("DOTV" + str(i) + self.name, [None], [None], "Dot V " + str(i)))  # TODO

            self.add_edge(self.get_node("Q" + str(i) + self.name), self.get_node("DOTQK" + str(i)+ self.name))
            self.add_edge(self.get_node("K" + str(i) + self.name), self.get_node("DOTQK" + str(i)+ self.name))
            self.add_edge(self.get_node("DOTQK"+ str(i) + self.name), self.get_node("SoftMax" + str(i) + self.name))
            self.add_edge(self.get_node("SoftMax" + str(i) + self.name), self.get_node("DOTV" + str(i) + self.name))
            self.add_edge(self.get_node("V" + str(i) + self.name), self.get_node("DOTV" + str(i) + self.name))
            self.add_edge(self.get_node("DOTV"+ str(i) + self.name), self.get_node("HeadOutput" + str(i) + self.name))

        # green
        self.add_node(OutputOP("Output" + self.name, [self.input_shape[1]]))
        self.add_output(self.get_node('Output' + self.name))

        # macs
        for i in range(self.num_heads):
            self.add_edge(self.get_node("HeadOutput" + str(i) + self.name), self.get_node("Output" + self.name))
  