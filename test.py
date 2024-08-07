import torch
import matplotlib.pyplot as plt
import networkx as nx
from nngraph.graph import Graph
from nngraph.visualizer import Visualizer

g = Graph()
g.visualize()

# lg = list(g.nodes())[0]
# pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    # pos = {n: (n, n) for n in g} 
    # nx.draw(lg, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=3)

# nx.draw(g, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=20, font_size=3)
# for node in g.nodes():
#     subgraph = node
#     if subgraph.number_of_nodes() > 0:
#         sub_pos =  nx.nx_agraph.graphviz_layout(subgraph, prog='dot')  # Position for subgraph nodes
#         nx.draw(subgraph, pos=sub_pos, with_labels=True, node_color='lightgreen', edge_color='black', node_size=20, font_size=3)

# plt.savefig(g.output_path+'/graph_test.png', dpi=400)

viz = Visualizer()      
viz.visualize()