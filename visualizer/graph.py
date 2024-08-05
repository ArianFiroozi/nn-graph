import torch
import graphviz
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from torchviz import make_dot
import torch.nn as nn
from enum import Enum

class LayerType(Enum):
    CONV1D=1
    CONV2D=2
    LINEAR=3

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
        self.outputs=[]
    
    def get_label(self):
        return self.label

    def add_input_name(self, name:str):
        self.inputs.append(name)

    def add_output_name(self, name:str):
        self.outputs.append(name)

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
        printable += "\nW" + str(self.size)
        return printable

class InputOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Input"):
        super().__init__(name, OperationType.INPUT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nX" + str(self.size)
        return printable

class OutputOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Output"):
        super().__init__(name, OperationType.OUTPUT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nX" + str(self.size)
        return printable

class Layer: # node
    def __init__(self, name:str, type:LayerType, label:str="OP"):
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
        self.outputs=[]
    
    def add_input_name(self, name:str):
        self.inputs.append(name)

    def add_output_name(self, name:str):
        self.outputs.append(name)

class Conv1dLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, input_length,
                stride=1, padding=1, dilation=1, bias=None, sparsity=0.0):
        super().__init__(LayerType.LINEAR)
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

    def get_torch(self, x):
        return nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )

    def _build_operations():
        pass

class LinearLayer(Layer):
    def __init__(self, in_features, out_features, input_length):
        super().__init__(LayerType.LINEAR)
        self.in_features=in_features
        self.out_features=out_features
        self.input_length=input_length

    def _build_operations():
        pass

class Graph():
    def __init__(self, file_path='./model1.pkl', output_operational_graph_path='./visualizer', excluded_params = ["weight"]):

        self.types = ['conv1d', 'conv2d', 'linear']

        self.graph_dict = {}
        self.file_path=file_path
        self.pkl_dump={}
        self.layer_names=[]
        self.excluded = excluded_para
        self.output_operational_graph_path=output_operational_graph_path
    
    def visualize(self):
        pass

    def _read_pkl(self):
        with open(self.file_path, 'rb') as file:
            self.pkl_dump:dict = pickle.load(file)
        
        assert "modules" in self.pkl_dump.keys()
        self.layer_names = self.pkl_dump["modules"]
        for name in self.layer_names:
            self.graph_dict[name] = ""

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

    def _build_operational_layer(self, name, input_size=14):
        dot = graphviz.Digraph('cluster_' + name)

        if self._get_layer_type(name) == 'conv1d':
            conv_layer = nn.Conv1d(
                in_channels=int(self.pkl_dump[name]['in_channels']),
                out_channels=int(self.pkl_dump[name]['out_channels']),
                kernel_size=int(self.pkl_dump[name]['kernel_size'][0]),
                stride=int(self.pkl_dump[name]['stride'][0]),
                padding=int(self.pkl_dump[name]['padding'][0])
            )
            x = self.pkl_dump[name]["_parameters"]["weight"].shape

            kernel_size = conv_layer.kernel_size[0]
            padding = conv_layer.padding[0]
            stride = conv_layer.stride[0]
            dilation = conv_layer.dilation[0]
            num_input_channels = conv_layer.in_channels
            num_output_channels = conv_layer.out_channels

            output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            num_macs = output_size * kernel_size * num_input_channels * num_output_channels

            dot.node('Input' + str(name), f'Input: x{str([num_input_channels, input_size])}', shape='box', style='filled', color='lightblue')
            dot.node('Weight' + str(name), 'Weight' + str([num_output_channels, kernel_size]), shape='box', style='filled', color='lightblue')
            dot.node('Kernel' + str(name), f'Kernel: w[{kernel_size}]', shape='box', style='filled', color='lightblue')
            dot.node('Padding' + str(name), 'Padding: ' + str(padding), shape='ellipse', style='filled', color='lightgrey')
            dot.node('Dilation' + str(name), 'Dilation: ' + str(dilation), shape='ellipse', style='filled', color='lightgrey')
            dot.node('Output' + str(name), f'Output: y{[num_output_channels, output_size]}', shape='box', style='filled', color='lightgreen')
            dot.edge('Input' + str(name), 'Padding' + str(name))
            dot.edge('Weight' + str(name), 'Kernel' + str(name))
            dot.edge('Kernel' + str(name), 'Dilation' + str(name))

            for i in range(num_output_channels):
                dot.node('Output_c' + str(i) + str(name), 'Output Channel ' + str(i) + ': y' + str([output_size]), shape='box', style='filled', color='lightgreen')
                for j in range(output_size):
                    mac_node_name = f'Mac{name}_{i}_{j}'
                    input_start = j * stride - padding
                    weight_start = 0 
                    weight_end = kernel_size - 1

                    dot.node(mac_node_name, f'MAC {i},{j}' + f'\nusing: x[{input_start}:{input_start + kernel_size}], w[{i}][{weight_start}:{weight_end}]'
                                , shape='box', style='filled', color='lightgrey')
                    
                    dot.edge('Padding' + str(name), mac_node_name)
                    dot.edge('Dilation' + str(name), mac_node_name)
                    dot.edge(mac_node_name, 'Output_c' + str(i) + str(name))
            
            for i in range(num_output_channels):
                dot.edge('Output_c' + str(i) + str(name), 'Output' + str(name))

        elif self._get_layer_type(name) == 'linear':
            linear_layer = nn.Linear(
                in_features=int(self.pkl_dump[name]['in_features']),
                out_features=int(self.pkl_dump[name]['out_features'])
            )
            x = self.pkl_dump[name]["_parameters"]["weight"].shape

            num_input_features = linear_layer.in_features
            num_output_features = linear_layer.out_features

            output_size = num_output_features

            dot.node('Input' + str(name), f'Input: x{str([num_input_features])}', shape='box', style='filled', color='lightblue')
            dot.node('Weight' + str(name), 'Weight: w' + str([num_output_features, num_input_features]), shape='box', style='filled', color='lightblue')
            dot.node('Output' + str(name), f'Output: y{[num_output_features]}', shape='box', style='filled', color='lightgreen')

            for i in range(num_output_features):
                # dot.node('Output_f' + str(i) + str(name), 'Output Feature ' + str(i) + ': y' , shape='box', style='filled', color='lightgreen')
                
                mac_node_name = f'MAC{name}_{i}'
                
                dot.node(mac_node_name, f'MAC {i}' + f'\nusing: x, w[{i}]', shape='box', style='filled', color='lightgrey')
                dot.edge('Weight' + str(name), mac_node_name)
                
                dot.edge('Input' + str(name), mac_node_name)
                dot.edge(mac_node_name, 'Output' + str(name))
                
            # for i in range(num_output_features):
            #     dot.edge('Output_f' + str(i) + str(name), 'Output' + str(name))

        return dot

    def _build_operational_graph(self, show_sublayers=True):
        dot = graphviz.Digraph()
        names = []

        for name in self.layer_names:
            if not show_sublayers and name.count("."):
                continue
                
            layer_torch = self._get_torch_layer(name)
            if layer_torch is not None:
                layer_dot = self._build_operational_layer(name)
                if layer_dot is None:
                    continue

                layer_dot.attr(name='cluster_'+str(name),
                            label=str(name),
                            style='dashed',
                            color='black', 
                            penwidth='2')
                dot.subgraph(layer_dot)

                if len(names):
                    dot.edge('Output' + names[-1], 'Input'+str(name))

                names.append(str(name))
            else:
                with dot.subgraph(name='cluster_'+name) as sub:
                    sub.attr(style='dashed', color='black', penwidth='2')
                    sub.node(name, shape='box', style='filled', color='lightgrey')
                    sub.node('Input' + name, label='Input',  shape='box', style='filled', color='lightblue')
                    sub.node('Output' + name, label='Output', shape='box', style='filled', color='lightgreen')
                    sub.edge('Input' + name, name)
                    sub.edge(name, 'Output' + name)
                    sub.attr(label='unknown op')

                if len(names):
                    dot.edge('Output' + names[-1], 'Input'+str(name))

                names.append(str(name))

        dot.render(self.output_operational_graph_path + '/operational_graph', format='png', cleanup=True) 
