import json

from scipy.spatial.distance import cosine

with open("data/jss/E-ELF-sim.json", "r") as f:
    graph_data = json.load(f)

nodes = graph_data["nodes"]["positions"]
radii = graph_data["info"]["nodes"]["radius"]
nodes = [[int(e) for e in p] for p in nodes]
# print(graph_data.keys())
with open("data/fuel_nodes.tsv", "w") as f:
    for node in nodes:
        i = nodes.index(node)
        w = radii[i]

        node = [str(e) for e in node]
        node = "\t".join(node)
        node += "\t" + str(w) + "\n"
        f.write(node)


thickness = graph_data["info"]["links"]["thickness"]
edges = list(graph_data["links"].keys())
with open("data/fuel_edges.tsv", "w") as f:
    for edge in graph_data["links"]:
        eps = graph_data["links"][edge]["end_points"]
        j = edges.index(edge)
        fromp = nodes[eps[0]]
        fromp = [str(e) for e in fromp]
        fromp = "\t".join(fromp)

        top = nodes[eps[1]]
        top = [str(e) for e in top]
        top = "\t".join(top)

        t = str(thickness[j])

        o = fromp + "\t" + top + "\t" + t + "\n"
        f.write(o)
