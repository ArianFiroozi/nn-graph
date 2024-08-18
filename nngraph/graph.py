import torch
from graphviz import Digraph
import pickle
import torch.nn as nn
from enum import Enum
import networkx as nx
from nngraph.layer import *
import json
from torchviz import make_dot

class Graph(nx.DiGraph):
    def __init__(self, input_pkl_path='./models/model3.pkl', output_path='./nngraph/outputs',
                config_file='./nngraph/layer_config.json', excluded_params = ["weight", "in_proj_weight", "in_proj_bias"],
                input_shape=[3,3]):
        super().__init__()

        self.input_pkl_path=input_pkl_path
        self.pkl_dump={}
        self.layer_names=[]
        self.excluded=excluded_params
        self.output_path=output_path
        self.input_shape=input_shape

        with open(config_file, 'r') as f:
            self.layer_config = json.load(f)

        self._read_pkl()
        self._build_graph()
    
    def add_layer(self, layer:Layer, append_latest=False):
        self.add_node(layer)
        if append_latest and self.__len__()>1:
            self.add_edge(layer, self.nodes[-1])

    def visualize(self, operational=True, layers=True, torch_functions=False):
        if operational:
            self._render_operational()
        if layers:
            self._render_layers()
        if torch_functions:
            self._render_torch_functions()

    def get_node(self, name:str)->Layer:
        for node in self.nodes():
            if node.get_name() == name:
                return node
        return None

    def _read_pkl(self):
        with open(self.input_pkl_path, 'rb') as file:
            self.pkl_dump:dict = pickle.load(file)
        
        assert "modules" in self.pkl_dump.keys()
        self.layer_names = self.pkl_dump["modules"]
        if "activations" in self.layer_names:
            self.layer_names.remove("activations")

    def _calc_sparsity(self, weights):
        total_elements = weights.numel()
        non_zero_elements = (abs(weights) != 0).sum().item()
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity

    def _get_layer_type(self, name):
        if 'type' in self.pkl_dump[name].keys():
            return self.pkl_dump[name]['type']
        elif 'in_features' in self.pkl_dump[name].keys():
            return 'linear'
        elif 'in_channels' in self.pkl_dump[name].keys():
            assert 'kernel_size' in self.pkl_dump[name].keys()
            if len(self.pkl_dump[name]['kernel_size']) == 2:
                return 'conv2d'
            elif len(self.pkl_dump[name]['kernel_size']) == 1:
                return 'conv1d'
        elif 'num_heads' in self.pkl_dump[name].keys():
            return 'attention'
        elif 'dim' in self.pkl_dump[name].keys(): #for activaions there is no way to know the exact type 
            if 'glu' in name:
                return 'glu'

        return None

    def _last_layer(self)->Layer:
        if self.__len__():
            return list(self.nodes())[-1]
        else:
            return None

    def get_output_shape(self):
        last_layer = self._last_layer()
        if last_layer is None:
            return self.input_shape
        else:
            return last_layer.get_output_shape()

    def _build_layer(self, name)->Layer:
        input_shape = self.get_output_shape()

        layer_type = self._get_layer_type(name)
        if layer_type not in self.layer_config:
            label = name + '\n'
            for k in self.pkl_dump[name].keys():
                if not k.startswith('_'):
                    label += str(k) + ": " + str(self.pkl_dump[name][k]) + "\n"
            return Layer(name, LayerType.UNKNOWN, label, input_shape=input_shape)

        config = self.layer_config[layer_type]
        params = {param: (self.pkl_dump[name][param]) for param in config['params']}

        if layer_type == 'conv1d':
            return Conv1dLayer(name, **params, input_length=input_shape, weight=self.pkl_dump[name]['_parameters']['weight'], input_shape=input_shape)
        elif layer_type == 'conv2d':
            return Conv2dLayer(name, **params, input_width=input_shape[0], input_height=input_shape[1], 
            weight=self.pkl_dump[name]['_parameters']['weight'], input_shape=input_shape)
        elif layer_type == 'linear':
            return LinearLayer(name, **params, weight=self.pkl_dump[name]['_parameters']['weight'], input_shape=input_shape)
        elif layer_type == 'attention':
            return MHAttentionLayer(name, input_shape, **params)
        elif layer_type == 'glu':
            return GluLayer(name, input_shape, **params)

    def _build_graph(self, show_sublayers=False):
        prev_layer = None
        for name in self.layer_names:
            if (not show_sublayers and name.count(".")):
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

    def _render_torch_functions(self, show_sublayers=True, show_func_attrs=False):
        dot = Digraph()
        names = []

        for name in self.layer_names:
            if show_sublayers or not name.count("."):
                layer_torch = self.get_node(name).get_torch()
                if layer_torch is not None:
                    layer_dot = make_dot(layer_torch, 'cluster_'+name, show_attrs=show_func_attrs)
                    layer_dot.attr(name='cluster_'+str(name), label=self._get_layer_type(name), style='filled', color='lightgray', penwidth='2')
                    dot.subgraph(graph=layer_dot)

                    input_nodes = [node.split(' ')[0] for node in layer_dot.body if not '->' in node and 'lightblue' in node]
                    node_names = [node.split(' ')[0] for node in layer_dot.body if not '->' in node]

                    if names:
                        dot.edge(names[-1][1:], name)

                    for i in input_nodes:
                        dot.node(name, f"{name}\ntype: {self._get_layer_type(name)}")
                        dot.edge(name, i[1:])
                    names.append(node_names[0])
                else:
                    if names:
                        dot.edge(names[-1][1:], name)
                    names.append("\t"+str(name))
        
        dot.render(self.output_path + '/torch_functions_graph', format='png', cleanup=True) 

    def _render_layers(self, show_sublayers=True):
        node_attr = dict(style='filled', shape='box', align='left', fontsize='20', ranksep='0.1', height='1', width='1', fontname='monospace', label='')
        layers_graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        layers_graph.attr(dpi='300')

        def get_layer_dict():
            layer_dict={}
            for name in self.layer_names:
                layer_dict[name] = str(name) + "\n\n"
                for key in self.pkl_dump[name].keys():
                    if isinstance(self.pkl_dump[name][key], dict):
                        for key2 in self.pkl_dump[name][key]:
                            if key2 not in self.excluded:
                                layer_dict[name] += str(key2) + ": " + str(self.pkl_dump[name][key][key2]) + "\n"
                            elif isinstance(self.pkl_dump[name][key][key2], torch.Tensor):
                                layer_dict[name] += str(key2) + ": " + str(list(self.pkl_dump[name][key][key2].shape)) + "\n"

                    else:
                        if key not in self.excluded:
                            layer_dict[name] += str(key) + ": " + str(self.pkl_dump[name][key]) + "\n"
            return layer_dict

        layer_dict=get_layer_dict()
        previous_layer_name = None
        for key in layer_dict.keys():
            if not show_sublayers and key.count("."):
                continue
            layers_graph.node(key, layer_dict[key])
            if previous_layer_name != None :
                layers_graph.edge(previous_layer_name, key)
            previous_layer_name = key

        layers_graph.render(self.output_path + "/layers_graph", format="png", cleanup=True)

    def __len__(self):
        return len(self.nodes)
