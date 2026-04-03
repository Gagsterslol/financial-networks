import networkx as nx
import numpy as np

# Contagion simulation in a financial network

# We built a MST representing the similarity between stocks. Now we want to simulate a shock event and see how it propagates through the network.

# Simple model: Each node has a shock value. When a node is shocked, it propagates a fraction of its shock to its neighbors proportional to the correlation weight of the edge connecting them. To do so, we will use BFS to traverse the tree and update the shock values of the nodes.
def simulate_shock(tree, start_node, magnitude):
    print(f"Shock event with magnitude {magnitude} occurred!")

    shocks = {node: 0 for node in tree.nodes()}
    shocks[start_node] = magnitude
    for u, v in nx.bfs_edges(tree, source=start_node):
        #weight is the distance in the MST, we need to convert it back to correlation
        d = tree[u][v]['weight']
        correlation = 1 - (d**2 / 2)
        
        #the shock on the neighbor 'v' is the shock on 'u' scaled by correlation
        shocks[v] = shocks[u] * correlation

    return shocks