import torch
import matplotlib.pyplot as plt
import networkx as nx
from nngraph.graph import Graph
from nngraph.visualizer import Visualizer

g = Graph()
g.visualize()
pos = nx.circular_layout(g)
# pos = {n: (n, n) for n in g} 
nx.draw(g, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=5)
plt.savefig(g.output_path+'/graph_test.png')

viz = Visualizer()      
viz.visualize()