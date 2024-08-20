import torch
from graphviz import Digraph
import pickle
import torch.nn as nn
from enum import Enum
import networkx as nx
from nngraph.layer import *
import json
from torchviz import make_dot
import onnx
import onnx2torch 
import pathlib

class Graph(nx.DiGraph):
    def __init__(self, input_model_path='./models/model.onnx', output_path='./nngraph/outputs',
                config_file='./nngraph/layer_config.json', excluded_params = ["weight", "in_proj_weight", "in_proj_bias"],
                input_shape=[3,3]):
        super().__init__()

        self.input_model_path=input_model_path
        self.model_onnx=None
        self.model_torch=None
        self.layer_names=[]
        self.output_path=output_path
        self.input_shape=input_shape

        with open(config_file, 'r') as f: # add if model has type
            self.layer_config = json.load(f)

        self._read_model()
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

    def _read_model(self):
        self.model_torch=onnx2torch.convert('models/model.onnx') ## TODO:wtf
        self.model_onnx=onnx.load('models/model.onnx')

        for node in self.model_onnx.graph.node:
            name = '/'.join(node.name.split('/')[1:-1])
            if name == "":
                name = node.name.split('/')[-1]
            if name not in self.layer_names:
                self.layer_names.append(name)

    def _calc_sparsity(self, weights):
        total_elements = weights.numel()
        non_zero_elements = (abs(weights) != 0).sum().item()
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity

    def _get_layer_type(self, name): # use if added type
        pass

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
        # input_shape = self.get_output_shape()
        return Layer(name, model_onnx=self.model_onnx, label=name)

    def __add_edges_between_layers(self):
        all_nodes={}
        for layer in self.nodes():
            for node in layer.nodes:
                all_nodes[node] =layer
        
        for node in all_nodes.keys():
            known_inputs = []
            for pred in all_nodes[node].predecessors(node):
                known_inputs+=pred.outputs
        
            for input_name in node.inputs: 
                if input_name not in known_inputs:
                    for other in all_nodes:
                        if input_name in other.outputs:
                            self.add_edge(all_nodes[node], all_nodes[other])
                            break

    def _build_graph(self, show_sublayers=True):
        for name in self.layer_names:           
            new_layer = self._build_layer(name)
            self.add_node(new_layer)

        self.__add_edges_between_layers()

    def _render_operational(self):
        dot = Digraph()
        for layer in self.nodes:
            dot.subgraph(layer.get_visual())

        for prev_layer in self.nodes: # clean this TODO:this should be done in graph
            for layer in self.nodes:
                if layer == prev_layer:
                    continue
                for output in prev_layer.outputs:
                    for input in layer.inputs:
                        for o in output.outputs:
                            for i in input.inputs:
                                if i == o:
                                    dot.edge(output.get_name(), input.get_name())
                                    break
        for edge in self.edges:
            print(edge)
            dot.edge(edge[1].name,edge[0].name)

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

    def _render_layers(self, show_sublayers=True): #TODO: add if layer file added
        pass 
        # node_attr = dict(style='filled', shape='box', align='left', fontsize='20', ranksep='0.1', height='1', width='1', fontname='monospace', label='')
        # layers_graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        # layers_graph.attr(dpi='300')

        # def get_layer_dict():
        #     layer_dict={}
        #     for name in self.layer_names:
        #         layer_dict[name] = str(name) + "\n\n"
        #         for key in self.pkl_dump[name].keys():
        #             if isinstance(self.pkl_dump[name][key], dict):
        #                 for key2 in self.pkl_dump[name][key]:
        #                     if key2 not in self.excluded:
        #                         layer_dict[name] += str(key2) + ": " + str(self.pkl_dump[name][key][key2]) + "\n"
        #                     elif isinstance(self.pkl_dump[name][key][key2], torch.Tensor):
        #                         layer_dict[name] += str(key2) + ": " + str(list(self.pkl_dump[name][key][key2].shape)) + "\n"

        #             else:
        #                 if key not in self.excluded:
        #                     layer_dict[name] += str(key) + ": " + str(self.pkl_dump[name][key]) + "\n"
        #     return layer_dict

        # layer_dict=get_layer_dict()
        # previous_layer_name = None
        # for key in layer_dict.keys():
        #     if not show_sublayers and key.count("."):
        #         continue
        #     layers_graph.node(key, layer_dict[key])
        #     if previous_layer_name != None :
        #         layers_graph.edge(previous_layer_name, key)
        #     previous_layer_name = key

        # layers_graph.render(self.output_path + "/layers_graph", format="png", cleanup=True)

    def __len__(self):
        return len(self.nodes)
