import igraph
from sklearn.preprocessing import MinMaxScaler

nodes = {}
nodelist = []
with open("data/nodes.tsv", "r") as f:
    for l in f:
        node, pr = l.strip().split("\t")
        pr = float(pr)
        nodes[node] = pr
        nodelist.append(node)

edges = []
with open("data/edges.tsv", "r") as f:
    for l in f:
        fromid, toid = l.strip().split("\t")
        if fromid in nodes and toid in nodes:
            t = (fromid, toid)
            edges.append(t)

g = igraph.Graph()
for n,p in nodes.items():
    g.add_vertex(n, pr=p)

for e in edges:
    g.add_edge(e[0], e[1])

#l = g.layout_fruchterman_reingold_3d()
l = g.layout_kamada_kawai_3d()
coords = []
for c in l:
    coords.append(c)

scaler = MinMaxScaler(feature_range=(-300, 300))
scaler.fit(coords)
rescaled_coords = scaler.transform(coords)

node_coord = {}
for i in range(len(nodelist)):
    node = nodelist[i]
    node_coords = rescaled_coords[i]
    node_coord[node] = list(rescaled_coords[i])

with open("data/node_coords.tsv", "w") as f:
    for i in range(len(nodelist)):
        node = nodelist[i]
        node_coords = list(rescaled_coords[i])
        node_coords = [str(i) for i in node_coords]
        node_coords = "\t".join(node_coords)
        radius = nodes[node]
        o = node_coords + "\t" + str(radius) + "\n"
        f.write(o)

with open("data/edge_coords.tsv", "w") as f:
    for e in edges:
        n1, n2 = e[0], e[1]
        coord1 = list(node_coord[n1])
        coord2 = list(node_coord[n2])
        coord1 = [str(i) for i in coord1]
        coord2 = [str(i) for i in coord2]
        coord1 = "\t".join(coord1)
        coord2 = "\t".join(coord2)
        o = coord1 + "\t" + coord2 + "\n"
        f.write(o)
