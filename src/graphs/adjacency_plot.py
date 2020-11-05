import colorsys

import community
import networkx as nx
import numpy as np
from PIL import Image
import termplotlib as tpl

G = nx.read_graphml("data/backbone10.graphml")
print(len(G.nodes), len(G.edges))

nodes = sorted(G.nodes)

nodes_labels = {}
for node in nodes:
    nodes_labels[node] = G.nodes[node]['label']
    G.nodes[node].pop('label', None)
# community detection
partition = community.best_partition(G)
bins = sorted(set(partition.values()))
counts = [len([v for v in partition.values() if v == b]) for b in bins]

fig = tpl.figure()
fig.barh(counts, bins, force_ascii=False)
fig.show()

# centrality measures
pr = nx.pagerank(G, weight='weight')
deg = nx.degree_centrality(G)

nx.set_node_attributes(G, partition, "partition")
nx.set_node_attributes(G, pr, "PageRank")
nx.set_node_attributes(G, deg, "Degree")
nx.set_node_attributes(G, nodes_labels, "label")
# save information
nx.write_graphml(G, "data/bb_partition.graphml")

# PageRank and degree centrality lists
pr = {k: v for k, v in sorted(pr.items(),
                              key=lambda item: item[1],
                              reverse=True)}
with open("data/tables/pagerank.tsv", "w") as outfile:
    h="Name\turl\tpartition\tpagerank\n"
    outfile.write(h)
    for k, v in pr.items():
        name = G.nodes[k]['label']
        if name:
            part = str(G.nodes[k]['partition'])
            o = name + "\t" + k + "\t" + part + "\t" + str(v) + "\n"
            outfile.write(o)
deg = {k: v for k, v in sorted(deg.items(),
                               key=lambda item: item[1],
                               reverse=True)}
with open("data/tables/degree.tsv", "w") as outfile:
    h="Name\turl\tpartition\tdegree\n"
    outfile.write(h)
    for k, v in pr.items():
        name = G.nodes[k]['label']
        if name:
            part = str(G.nodes[k]['partition'])
            o = name + "\t" + k + "\t" + part + "\t" + str(v) + "\n"
            outfile.write(o)

# dotplot
sorted_nodes = []
grid_positions = [0]
pos = 0
for i in range(0, max(partition.values())):
    partition_nodes = [n for n in nodes if partition[n] == i]
    pos += len(partition_nodes)
    grid_positions.append(pos)
    sorted_nodes.extend(partition_nodes)
print(len(sorted_nodes))
edge_data = {}

for edge in G.edges:
    p1, p2 = partition[edge[0]], partition[edge[1]]
    if edge[0] in sorted_nodes and edge[1] in sorted_nodes:
        w = G.edges[edge]['weight']
        w = min(abs(100-w), 100)
        if p1 == p2:
            t = colorsys.hls_to_rgb(346, w, 66)
        else:
            t = colorsys.hls_to_rgb(68, w, 91)
        t = np.asarray(t, dtype=np.uint8)
        i = sorted_nodes.index(edge[0])
        j = sorted_nodes.index(edge[1])
        v = (i, j)
        edge_data[v] = t

size = len(sorted_nodes)
points = []
default_color = np.asarray([255, 255, 255], dtype=np.uint8)
for i in range(size):
    row = []
    for j in range(size):
        if (i, j) in edge_data:
            row.append(edge_data[(i, j)])
        else:
            row.append(default_color)
    row = np.asarray(row, dtype=np.uint8)
    points.append(row)


points = np.asarray(points, dtype=np.uint8)
im = Image.fromarray(np.uint8(points), "RGB")
im.save("vizs/theviz.png", "PNG")

ázasság 