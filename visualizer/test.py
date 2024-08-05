from graphviz import Digraph

# Create a new directed graph
dot = Digraph()

# Create the first subgraph
with dot.subgraph(name='cluster_1') as sub:
    sub.attr(label='Subgraph 1')
    sub.node('A', 'Node A')
    sub.node('B', 'Node B')

# Create the second subgraph
with dot.subgraph(name='cluster_2') as sub:
    sub.attr(label='Subgraph 2')
    sub.node('C', 'Node C')
    sub.node('D', 'Node D')

# Add edges between the subgraphs
dot.edge('cluster_1', 'cluster_2', label='Edge from Subgraph 1 to Subgraph 2')
dot.render('./IR/visualizer/test', format='png', cleanup=True)