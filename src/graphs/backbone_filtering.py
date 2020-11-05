import networkx as nx

from src.graphs.disparity import calc_alpha_ptile, cut_graph, disparity_filter

G = nx.read_graphml("data/persons_full.graphml")
print(len(G.nodes), len(G.edges))

alpha_measures = disparity_filter(G)
quantiles, num_quant = calc_alpha_ptile(alpha_measures)
alpha_cutoff = quantiles[5]

cut_graph(G, alpha_cutoff, 10)

nx.write_graphml(G, "data/backbone10.graphml")
print(len(G.nodes), len(G.edges))
