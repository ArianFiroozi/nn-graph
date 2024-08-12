import torch
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import torch.nn as nn
from enum import Enum
import networkx as nx
from nngraph.layer import *

class Graph(nx.DiGraph):
    def __init__(self, input_pkl_path='./models/model_big.pkl', output_path='./nngraph/outputs', excluded_params = ["weight"]):
        super().__init__()
        self.types = ['conv1d', 'conv2d', 'linear']

        self.input_pkl_path=input_pkl_path
        self.pkl_dump={}
        self.layer_names=[]
        self.excluded=excluded_params
        self.output_path=output_path


        self._read_pkl()
        self._build_graph()
    
    def visualize(self):
        self._render_operational()

    def _read_pkl(self):
        with open(self.input_pkl_path, 'rb') as file:
            self.pkl_dump:dict = pickle.load(file)
        
        assert "modules" in self.pkl_dump.keys()
        self.layer_names = self.pkl_dump["modules"]

    def _calc_sparsity(self, weights):
        total_elements = weights.numel()
        non_zero_elements = (abs(weights) != 0).sum().item()
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity

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
        elif 'num_heads' in self.pkl_dump[name].keys():
            type = 'attention'

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

        elif type == 'attention':
            attention_layer = nn.MultiheadAttention(self.pkl_dump[name]['embed_dim'], 
                                                self.pkl_dump[name]['num_heads'],
                                                self.pkl_dump[name]['dropout'])

            sequence_length = 3 
            embed_dim = self.pkl_dump[name]['embed_dim']
    
            x = torch.randn(sequence_length, embed_dim)
            y, attn_output_weights = attention_layer(x, x, x)

        else:
            print("nngraph: unknown layer type->"+name)
            return None

        return y

    #TODO: external layer addition
    def _build_layer(self, name, input_len=3)->Layer: #TODO: add output shapes for each layer
        if self._get_layer_type(name) == 'conv1d':
            in_channels=int(self.pkl_dump[name]['in_channels'])
            out_channels=int(self.pkl_dump[name]['out_channels'])
            kernel_size=list(self.pkl_dump[name]['kernel_size'])
            stride=list(self.pkl_dump[name]['stride'])
            padding=list(self.pkl_dump[name]['padding'])
            dilation=[1] # change if added

            return Conv1dLayer(name, in_channels, out_channels, kernel_size,
                                input_len, stride, padding, dilation, weight=self.pkl_dump[name]['_parameters']['weight']) # may add sparsity

        elif self._get_layer_type(name) == 'conv2d':
            in_channels=int(self.pkl_dump[name]['in_channels'])
            out_channels=int(self.pkl_dump[name]['out_channels'])
            kernel_size=list(self.pkl_dump[name]['kernel_size'])
            stride=list(self.pkl_dump[name]['stride'])
            padding=list(self.pkl_dump[name]['padding'])
            dilation=[1, 1] # change if added

            return Conv2dLayer(name, in_channels, out_channels, kernel_size,
                                input_len, input_len, stride, padding, dilation, weight=self.pkl_dump[name]['_parameters']['weight']) # may add sparsity

        elif self._get_layer_type(name) == 'linear':
            in_features = int(self.pkl_dump[name]['in_features'])
            out_features = int(self.pkl_dump[name]['out_features'])

            return LinearLayer(name, in_features, out_features)

        elif self._get_layer_type(name) == 'attention':
            embed_dim = int(self.pkl_dump[name]['embed_dim'])
            num_heads = int(self.pkl_dump[name]['num_heads'])
            print(self.pkl_dump[name])
            return MHAttentionLayer(name, [input_len, 8], embed_dim, num_heads) # fix if output added
            
        else:
            label = name + '\n'
            for k in self.pkl_dump[name].keys():
                if not k.startswith('_'):
                    label += str(k) + ": " + str(self.pkl_dump[name][k]) + "\n"
            return Layer(name, LayerType.UNKNOWN, label)

    def _build_graph(self, show_sublayers=True):
        prev_layer = None
        for name in self.layer_names:
            if not show_sublayers and name.count("."):
                continue

            new_layer = self._build_layer(name)
            self.add_node(new_layer)

            if self.__len__()>1:
                self.add_edge(prev_layer, new_layer)
            prev_layer=new_layer

    def _render_operational(self):
        dot = Digraph()
        for layer in self.nodes:
            dot.subgraph(layer.get_visual())

        prev_layer = None
        for layer in self.nodes:
            if prev_layer==None:
                prev_layer=layer
                continue

            for output in prev_layer.outputs: # fix for multiple outputs
                for input in layer.inputs:
                    dot.edge(output.get_name(), input.get_name())
            prev_layer=layer

        dot.render(self.output_path + '/operational_graph', format='png', cleanup=True) 

    def __len__(self):
        return len(self.nodes)
