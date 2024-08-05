import torch
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import torch.nn as nn
from enum import Enum

class LayerType(Enum):
    CONV1D=1
    CONV2D=2
    LINEAR=3
    UNKNOWN=0

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
    UNKNOWN=0

class Operation: # node
    def __init__(self, name:str, type:OperationType=0, label:str="OP"):
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
    
    def get_label(self):
        return self.label

    def add_input_name(self, name:str):
        self.inputs.append(name)
    
    def get_name(self)->str:
        return self.name

class PaddOP(Operation):
    def __init__(self, name:str, padding:int, label:str="Padding"):
        super().__init__(name, OperationType.PADD, label)
        self.padding = padding

    def get_label(self):
        return self.label + ": " + str(self.padding)

class DilationOP(Operation):
    def __init__(self, name:str, dilation:int, label:str="Dilation"):
        super().__init__(name, OperationType.DILATION, label)
        self.dilation = dilation

    def get_label(self):
        return self.label + ": " + str(self.dilation)

class MacOP(Operation):
    def __init__(self, name:str, input_index, weight_index, label:str="MAC"):
        super().__init__(name, OperationType.MAC, label)
        self.input_index=input_index
        self.weight_index=weight_index

    def get_label(self):
        printable = self.label + ":" 
        printable += "\ninp" + str(self.input_index) 
        printable += "\nweight" + str(self.weight_index)
        return printable

class ConstOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Const"):
        super().__init__(name, OperationType.CONST, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nW" + str(self.shape)
        return printable

class InputOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Input"):
        super().__init__(name, OperationType.INPUT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nX" + str(self.shape)
        return printable

class OutputOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Output"):
        super().__init__(name, OperationType.OUTPUT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nX" + str(self.shape)
        return printable

class Layer: # node
    def __init__(self, name:str, type:LayerType, label:str="OP", build_dummy_op=True):
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
        self.outputs=[]
        self.nodes=[]
        if build_dummy_op:
            self._build_operations()
    
    def add_input_name(self, name:str):
        self.inputs.append(name)

    def add_output_name(self, name:str):
        self.outputs.append(name)
    
    def add_node(self, node:Operation):
        self.nodes.append(node)

    def get_node(self, name:str)->Operation:
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def get_name(self)->str:
        return self.name

    def add_edge(self, begin, end):
        for i in range(len(self.nodes)):
            if self.nodes[i].get_name() == end:
                self.nodes[i].add_input_name(begin)
                return
    
    def _build_operations(self):
        self.add_node(InputOP('Input'+self.name,[-1]))
        self.add_input_name(self.nodes[-1].get_name())
        self.add_node(Operation('OP'+self.name, label=self.name))
        self.add_node(OutputOP('Output'+self.name,[-1]))
        self.add_output_name(self.nodes[-1].get_name())

        self.add_edge(self.nodes[0].get_name(), self.nodes[1].get_name())
        self.add_edge(self.nodes[1].get_name(), self.nodes[2].get_name())

    def get_visual(self):
        dot = Digraph('cluster_' + self.name)
        dot.attr(label=self.label,
                style='dashed',
                color='black', 
                penwidth='2')

        for node in self.nodes:
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
        return dot

class Conv1dLayer(Layer):
    def __init__(self, name:str, in_channels, out_channels, kernel_size, input_length,
                stride=1, padding=1, dilation=1, bias=None, sparsity=0.0, label="Conv1d"):
        super().__init__(name, LayerType.LINEAR, label, False)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sparsity = sparsity
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias is None else bias
        
        self._build_operations()

    def get_torch(self, x):
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
        self.add_edge(self.nodes[0].get_name(), self.nodes[1].get_name())
        self.add_node(InputOP("Input" + self.name, [self.in_channels, self.input_length]))
        self.add_input_name(self.nodes[-1].get_name())

        #grey
        self.add_node(PaddOP("Padding" + self.name, self.padding))
        self.add_node(DilationOP("Dilation" + self.name, self.dilation))
        self.add_edge("Kernel" + self.name, "Dilation" + self.name)
        self.add_edge("Input" + self.name, "Padding" + self.name)

        #green
        self.add_node(OutputOP("Output" + self.name, [self.out_channels, self._calc_out_size()]))
        self.add_output_name(self.nodes[-1].get_name())

        #macs
        for i in range(self.out_channels):
            self.add_node(OutputOP('Output_c' + str(i) + self.name, [self._calc_out_size()], "Output Channel"))
            for j in range(self._calc_out_size()):
                mac_node_name = f'Mac{self.name}_{i}_{j}'
                input_start = j * self.stride - self.padding
                weight_start = 0 
                weight_end = self.kernel_size - 1

                self.add_node(MacOP(mac_node_name, [input_start, input_start+self.kernel_size],
                                [weight_start, weight_end], "MAC" + str(i) + "," + str(j)))                
                self.add_edge('Padding' + self.name, mac_node_name)
                self.add_edge('Dilation' + self.name, mac_node_name)
                self.add_edge(mac_node_name, 'Output_c' + str(i) + self.name)
        
        for i in range(self.out_channels):
            self.add_edge('Output_c' + str(i) + self.name, 'Output' + self.name)

    def _calc_out_size(self):
        return (self.input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

class LinearLayer(Layer):
    def __init__(self, in_features, out_features, input_length):
        super().__init__(LayerType.LINEAR)
        self.in_features=in_features
        self.out_features=out_features
        self.input_length=input_length

    def _build_operations():
        pass

class Graph():
    def __init__(self, file_path='./model2.pkl', output_path='./visualizer', excluded_params = ["weight"]):

        self.types = ['conv1d', 'conv2d', 'linear']

        self.graph_dict={}
        self.file_path=file_path
        self.pkl_dump={}
        self.layer_names=[]
        self.excluded=excluded_params
        self.output_path=output_path
        self.layers=[] #ordered

        self._read_pkl()
        self._read_layers()
        self._build_graph()
    
    def visualize(self):
        self._render_operational()

    def _read_pkl(self):
        with open(self.file_path, 'rb') as file:
            self.pkl_dump:dict = pickle.load(file)
        
        assert "modules" in self.pkl_dump.keys()
        self.layer_names = self.pkl_dump["modules"]
        for name in self.layer_names:
            self.graph_dict[name] = ""

    def _calc_sparsity(self, weights):
        total_elements = weights.numel()
        non_zero_elements = (abs(weights) != 0).sum().item() # fix
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity

    def _read_layers(self):
        for name in self.layer_names:
            self.graph_dict[name] += str(name) + "\n\n"
            for key in self.pkl_dump[name].keys():
                if isinstance(self.pkl_dump[name][key], dict):
                    for key2 in self.pkl_dump[name][key]:
                        if key2 not in self.excluded:
                            self.graph_dict[name] += str(key2) + ": " + str(self.pkl_dump[name][key][key2]) + "\n"
                else:
                    if key not in self.excluded:
                        self.graph_dict[name] += str(key) + ": " + str(self.pkl_dump[name][key]) + "\n"
            
            if "_parameters" in self.pkl_dump[name] and "weight" in self.pkl_dump[name]["_parameters"]:
                self.graph_dict[name] += "sparsity: " + str(self._calc_sparsity(self.pkl_dump[name]["_parameters"]["weight"]))

    def _get_layer_type(self, name):
        type = None
        if 'type' in self.pkl_dump[name].keys():
            type = self.pkl_dump[name]['type']
        elif 'in_features' in self.pkl_dump[name].keys():
            type = 'linear'
        elif 'in_channels' in self.pkl_dump[name].keys():
            assert 'kernel_size' in self.pkl_dump[name].keys()
            if len(self.pkl_dump[name]['kernel_size']) == 2:
                type = 'conv2d'
            elif len(self.pkl_dump[name]['kernel_size']) == 1:
                type = 'conv1d'

        return type

    def _get_torch_layer(self, name)->torch.nn:
        type = self._get_layer_type(name)

        if type == 'linear':
            linear_layer = nn.Linear(int(self.pkl_dump[name]['in_features']), int(self.pkl_dump[name]['out_features']))
            x = torch.randn(1, self.pkl_dump[name]['in_features'])
            y = linear_layer(x)

        elif type == 'conv2d':
            conv_layer = nn.Conv2d(
                in_channels=int(self.pkl_dump[name]['in_channels']),
                out_channels=int(self.pkl_dump[name]['out_channels']),
                kernel_size=int(self.pkl_dump[name]['kernel_size'][0]),
                stride=int(self.pkl_dump[name]['stride'][0]),
                padding=int(self.pkl_dump[name]['padding'][0])
            )
            x = torch.randn(self.pkl_dump[name]["_parameters"]["weight"].shape)
            y = conv_layer(x)

        elif type == 'conv1d':
            conv_layer = nn.Conv1d(
                in_channels=int(self.pkl_dump[name]['in_channels']),
                out_channels=int(self.pkl_dump[name]['out_channels']),
                kernel_size=int(self.pkl_dump[name]['kernel_size'][0]),
                stride=int(self.pkl_dump[name]['stride'][0]),
                padding=int(self.pkl_dump[name]['padding'][0])
            )
            x = torch.randn(self.pkl_dump[name]["_parameters"]["weight"].shape)
            y = conv_layer(x)

        else:
            print("visualizer: unknown layer type->"+name)
            return None

        return y

    def _build_layer(self, name, input_len=4):
        layer = None
        if self._get_layer_type(name) == 'conv1d':
            torch_layer = nn.Conv1d(
                in_channels=int(self.pkl_dump[name]['in_channels']),
                out_channels=int(self.pkl_dump[name]['out_channels']),
                kernel_size=int(self.pkl_dump[name]['kernel_size'][0]),
                stride=int(self.pkl_dump[name]['stride'][0]),
                padding=int(self.pkl_dump[name]['padding'][0])
            )
            kernel_size = torch_layer.kernel_size[0]
            padding = torch_layer.padding[0]
            stride = torch_layer.stride[0]
            dilation = torch_layer.dilation[0]
            num_input_channels = torch_layer.in_channels
            num_output_channels = torch_layer.out_channels
            layer = Conv1dLayer(name, torch_layer.in_channels, torch_layer.out_channels,
                            kernel_size, input_len, stride, padding, dilation) # may add sparsity

        # elif self._get_layer_type(name) == 'linear':
        #     linear_layer = nn.Linear(
        #         in_features=int(self.pkl_dump[name]['in_features']),
        #         out_features=int(self.pkl_dump[name]['out_features'])
        #     )
        #     x = self.pkl_dump[name]["_parameters"]["weight"].shape

        #     num_input_features = linear_layer.in_features
        #     num_output_features = linear_layer.out_features

        #     output_size = num_output_features

        #     dot.node('Input' + str(name), f'Input: x{str([num_input_features])}', shape='box', style='filled', color='lightblue')
        #     dot.node('Weight' + str(name), 'Weight: w' + str([num_output_features, num_input_features]), shape='box', style='filled', color='lightblue')
        #     dot.node('Output' + str(name), f'Output: y{[num_output_features]}', shape='box', style='filled', color='lightgreen')

        #     for i in range(num_output_features):              
        #         mac_node_name = f'MAC{name}_{i}'
                
        #         dot.node(mac_node_name, f'MAC {i}' + f'\nusing: x, w[{i}]', shape='box', style='filled', color='lightgrey')
        #         dot.edge('Weight' + str(name), mac_node_name)
                
        #         dot.edge('Input' + str(name), mac_node_name)
        #         dot.edge(mac_node_name, 'Output' + str(name))
        else:
            layer = Layer(name, LayerType.UNKNOWN, name)

        return layer

    def _build_graph(self, show_sublayers=True):
        for name in self.layer_names:
            if not show_sublayers and name.count("."):
                continue

            new_layer = self._build_layer(name)
            self.layers.append(new_layer)

    def _render_operational(self):
        dot = Digraph()
        for layer in self.layers:
            dot.subgraph(layer.get_visual())

        prev_layer = None
        for layer in self.layers:
            if prev_layer==None:
                prev_layer=layer
                continue

            for output in prev_layer.outputs: # fix for multiple outputs
                for input in layer.inputs:
                    dot.edge(output, input)
            prev_layer=layer

        dot.render(self.output_path + '/new_operational_graph', format='png', cleanup=True) 

g = Graph()
g.visualize()