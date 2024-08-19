import torch
import matplotlib.pyplot as plt
import networkx as nx
from nngraph.graph import Graph
import numpy as np
import onnx
from onnx2torch import convert as cm
import google

# model = onnx.load('models/model.onnx')

# print(model.graph)
model = cm("models/model.onnx")
# for i ,n in model.named_modules():
#     print(i, n.type)
# print(list(model.named_modules())[0][1])

from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnx
import os 

model = onnx.load("models/model.onnx")
# for i in model:
#     print(type(i))
# for i in model.ListFields():
#     print(i[1])
# for n in model.ListFields()[3][1].node:
#     print(n)
# print(onnx.helper.printable_graph(model.graph))
for i in model.graph.input:
    print(i)



pydot_graph = GetPydotGraph(
    model.graph,
    name=model.graph.name,
    rankdir="TB",
    node_producer=GetOpNodeProducer("docstring"),
)
pydot_graph.write_dot("graph2.dot")
os.system("dot -O -Tpng graph2.dot")