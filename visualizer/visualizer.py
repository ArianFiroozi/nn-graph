import torch
import graphviz
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from torchviz import make_dot
import torch.nn as nn

class Visualizer():
    def __init__(self, file_path='./model3.pkl', output_graph_path="./visualizer/outputs", 
                output_operational_graph_path='./visualizer/outputs', output_func_graph_path='./visualizer/outputs', excluded_params = ["weight"], threshold=100):
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
        self.output_operational_graph_path=output_operational_graph_path
    
    def visualize(self, functions_graph=True, show_sublayers=True, show_func_attrs=False, display_after=False):
        self._read_pkl()
        self._read_layers()
        self._build_graph(show_sublayers)

        if functions_graph:
            self._build_functions_graph(show_sublayers, show_func_attrs)

        self._build_operational_graph()

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
                        else:
                            self.graph_dict[name] += str(key2) + ": " + str(self.pkl_dump[name][key][key2].shape) + "\n"

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
                            + 'type: ' + self._get_layer_type(name) + '\n'
                            + str(self.pkl_dump[name]['_parameters']['weight'].shape))
                    dot.edge(name, i[1:])

                names.append(node_names[0])
                
            else:
                if len(names):
                    dot.edge(names[-1][1:], name)

                names.append("\t"+str(name))
        
        dot.render(self.output_func_graph_path + '/functions_graph', format='png', cleanup=True) 

    def _build_operational_layer(self, name, input_size=4):
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

viz = Visualizer()      
viz.visualize()
