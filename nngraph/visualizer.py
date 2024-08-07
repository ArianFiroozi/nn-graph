import torch
import graphviz
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from torchviz import make_dot
import torch.nn as nn

class Visualizer():
    def __init__(self, file_path='./models/model_big.pkl', output_graph_path="./nngraph/outputs", 
                output_func_graph_path='./nngraph/outputs', excluded_params = ["weight", "bias", "in_proj_weight", "in_proj_bias"], threshold=100):
        torch.set_printoptions(threshold=threshold)
        node_attr = dict(style='filled',
                            shape='box',
                            align='left',
                            fontsize='20',
                            ranksep='0.1',
                            height='1',
                            width='1',
                            fontname='monospace',
                            label='')
        edge_attr = dict(label="")
        self.types = ['conv1d', 'conv2d', 'linear']

        self.nn_graph = graphviz.Digraph(node_attr=node_attr, edge_attr=edge_attr, graph_attr=dict(size="12,12"))
        self.graph_dict = {}
        self.file_path=file_path
        self.pkl_dump={}
        self.layer_names=[]
        self.excluded = excluded_params
        self.nn_graph.attr(dpi='300')
        self.output_graph_path=output_graph_path
        self.output_func_graph_path=output_func_graph_path
    
    def visualize(self, functions_graph=True, show_sublayers=True, show_func_attrs=False, display_after=False):
        self._read_pkl()
        self._read_layers()
        self._build_graph(show_sublayers)

        if functions_graph:
            self._build_functions_graph(show_sublayers, show_func_attrs)

        if display_after:
            self.dispay_img()

    def _read_pkl(self):
        with open(self.file_path, 'rb') as file:
            self.pkl_dump:dict = pickle.load(file)
        
        assert "modules" in self.pkl_dump.keys()
        self.layer_names = self.pkl_dump["modules"]
        for name in self.layer_names:
            self.graph_dict[name] = ""
        # print(self.pkl_dump)

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
                        elif isinstance(self.pkl_dump[name][key][key2], torch.Tensor):
                            self.graph_dict[name] += str(key2) + ": " + str(list(self.pkl_dump[name][key][key2].shape)) + "\n"

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
        elif 'num_heads' in self.pkl_dump[name].keys():
            type ='attention' 

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

            sequence_length = 3  # Example sequence length
            batch_size = 1        # Example batch size
            embed_dim = self.pkl_dump[name]['embed_dim']
    
            x = torch.randn(sequence_length, embed_dim)
            y, attn_output_weights = attention_layer(x, x, x)
            print(attn_output_weights.shape, embed_dim)    

        else:
            print("visualizer: unknown layer type->"+name)
            return None

        return y

    def _build_functions_graph(self, show_sublayers=True, show_func_attrs=False):
        dot = graphviz.Digraph()
        names = []

        for name in self.layer_names:
            if not show_sublayers and name.count("."):
                continue
                
            layer_torch = self._get_torch_layer(name)
            if layer_torch is not None:
                layer_dot = make_dot(layer_torch, 'cluster_'+name, show_attrs=show_func_attrs)
                layer_dot.attr(name='cluster_'+str(name),
                            label=self._get_layer_type(name),
                            style='filled',
                            color='lightgray', 
                            penwidth='2')
                dot.subgraph(graph=layer_dot)

                subgraph_nodes = [node for node in layer_dot.body if not '->' in node]  # only nodes not edges

                node_names = []
                input_nodes = []
                for node in subgraph_nodes:
                    node_name = node.split(' ')[0]  # extract the node name
                    if 'lightblue' in node:
                        input_nodes.append(node_name)
                    node_names.append(node_name)

                if len(names):
                    dot.edge(names[-1][1:], name)

                for i in input_nodes:
                    dot.node(name, 
                            str(name) + '\n' 
                            + 'type: ' + self._get_layer_type(name))
                    dot.edge(name, i[1:])

                names.append(node_names[0])
                
            else:
                if len(names):
                    dot.edge(names[-1][1:], name)

                names.append("\t"+str(name))
        
        dot.render(self.output_func_graph_path + '/functions_graph', format='png', cleanup=True) 

    def _build_graph(self, show_sublayers=True):
        for key in self.graph_dict.keys():
            if not show_sublayers and key.count("."):
                continue
            self.nn_graph.node(key, self.graph_dict[key])

        previous_layer_name = None
        for name in self.layer_names:
            # print(self.pkl_dump[name])
            if not show_sublayers and name.count("."):
                continue

            if previous_layer_name != None :
                self.nn_graph.edge(previous_layer_name, name)
            previous_layer_name = name

        self.nn_graph.render(self.output_graph_path + "/graph", format="png")

    def dispay_img(self):
        img = mpimg.imread(self.output_graph_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
