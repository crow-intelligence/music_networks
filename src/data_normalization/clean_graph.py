import operator

import networkx as nx
from sklearn.preprocessing import minmax_scale

G = nx.read_graphml("data/ppl.graphml")
G_filtered = nx.Graph()

giant = list(max(nx.connected_components(G), key=len))
for node in giant:
    G_filtered.add_node(node)

for e in G.edges:
    if e[0] in giant and e[1] in giant:
        G_filtered.add_edge(e[0], e[1])

pr = nx.pagerank(G_filtered)
sorted_pr = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)[:100]
final_nodes = [e[0] for e in sorted_pr]
prs = [e[1] for e in sorted_pr]
prs_rescaled = minmax_scale(prs, [0, 1])

with open("data/nodes.tsv", "w") as f:
    for i in range(len(sorted_pr)):
        node = sorted_pr[i][0]
        v = sorted_pr[i][1]
        o = node + "\t" + str(v) + "\n"
        f.write(o)

with open("data/edges.tsv", "w") as f:
    for e in G_filtered.edges:
        if e[0] in final_nodes and e[1] in final_nodes:
            o = e[0] + "\t" + e[1] + "\n"
            f.write(o)
