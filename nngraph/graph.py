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
    def __init__(self, input_model_path='./models/modelconv2d.onnx', output_path='./nngraph/outputs',
                config_file='./nngraph/layer_config.json', excluded_params = ["weight", "in_proj_weight", "in_proj_bias"],
                input_shape=[3,3]):
        super().__init__()

        self.input_model_path=input_model_path
        self.model_onnx=None
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

    def visualize(self, operational=True, layers=True):
        if operational:
            self._render_operational()
        if layers:
            self._render_layers()

    def get_node(self, name:str)->Layer:
        for node in self.nodes():
            if node.get_name() == name:
                return node
        return None

    def _read_model(self):
        self.model_onnx=onnx.load(self.input_model_path) ## TODO:wtf
        
        for node in self.model_onnx.graph.node:
            name = '/'.join(node.name.split('/')[1:-1])
            if name == "":
                name = node.name.split('/')[-1]
            if name not in self.layer_names:
                self.layer_names.append(name)

    def _get_layer_type(self, name): # use if added type
        pass

    def _build_layer(self, name)->Layer:
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

        for prev_layer in self.nodes: # this is a bad view, layers are connected not the ops
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
                                
        dot.render(self.output_path + '/operational_graph', format='png', cleanup=True) 

    def _render_layers(self, show_sublayers=True): #TODO: complete if layer file added
        dot = Digraph()
        for edge in self.edges:
            dot.edge(edge[1].name,edge[0].name)

        dot.render(self.output_path + '/layers_graph', format='png', cleanup=True) 

    def __len__(self):
        return len(self.nodes)
