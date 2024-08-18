import torch
import matplotlib.pyplot as plt
import networkx as nx
from nngraph.graph import Graph
import numpy as np
import onnx
from onnx2torch import convert as cm

model = onnx.load('models/model.onnx')

# print(model.graph)
# pytorch_model = cm("models/model_big.onnx")

print(model.graph)
# for name, layer in pytorch_model.named_children():
#     print(name, layer)

# from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# import onnx
# import os 

# model = onnx.load("models/model.onnx")

# ###################################
# # We convert it into a graph.
# pydot_graph = GetPydotGraph(
#     model.graph,
#     name=model.graph.name,
#     rankdir="TB",
#     node_producer=GetOpNodeProducer("docstring"),
# )
# pydot_graph.write_dot("graph2.dot")
# os.system("dot -O -Tpng graph2.dot")

# g = Graph()
# g.visualize()

# # lg = list(g.nodes())[0]
# # pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
#     # pos = {n: (n, n) for n in g} 
#     # nx.draw(lg, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=3)

# # nx.draw(g, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=20, font_size=3)
# # for node in g.nodes():
# #     subgraph = node
# #     if subgraph.number_of_nodes() > 0:
# #         sub_pos =  nx.nx_agraph.graphviz_layout(subgraph, prog='dot')  # Position for subgraph nodes
# #         nx.draw(subgraph, pos=sub_pos, with_labels=True, node_color='lightgreen', edge_color='black', node_size=20, font_size=3)

# # plt.savefig(g.output_path+'/graph_test.png', dpi=400)   