import json
import os
from shutil import copy2, rmtree

import community
import networkx as nx
from networkx.readwrite import json_graph

G = nx.read_graphml("data/persons_full.graphml")

partition = community.best_partition(G)


# weighted degree
degree_centrality = nx.degree_centrality(G)
degree_centrality = {k: v for k, v in sorted(degree_centrality.items(),
                                             key=lambda item: item[1],
                                             reverse=True)}

with open("data/tables/degree.tsv", "w") as outfile:
    h = "Személy\tFokszám\tOsztály\n"
    outfile.write(h)
    for k,v in degree_centrality.items():
        p = G.nodes[k]['label']
        if p:
            pt = partition[k]
            o = p + "\t" + str(v) + "\t" + str(pt) + "\n"
            outfile.write(o)


# pagerank
pagerank = nx.pagerank(G, weight='weight')
pagerank = {k: v for k, v in sorted(pagerank.items(),
                                    key=lambda item: item[1],
                                    reverse=True)}

with open("data/tables/pagerank.tsv", "w") as outfile:
    h = "Személy\tPageRank\tOsztály\n"
    outfile.write(h)
    for k,v in pagerank.items():
        p = G.nodes[k]['label']
        if p:
            pt = partition[k]
            o = p + "\t" + str(v) + "\t" + str(pt) + "\n"
            outfile.write(o)

for node in list(pagerank.keys())[:60]:
    person_name = G.nodes[node]['label']
    if person_name and not person_name.endswith(".html"):
        person_name = person_name.strip().replace(" ", "_")
        fname = person_name + ".graphml"
        gfname = person_name + ".json"
        E = nx.ego_graph(G, node, radius=1)
        E2 = nx.Graph()
        Ego = nx.Graph()

        nodes = [n for n in G.nodes if n in E.nodes]
        names = [G.nodes[n]['label'] for n in nodes]
        node_names = dict(zip(nodes, names))
        pr_mapping = {k: v for k, v in pagerank.items() if k in E.nodes}
        pr_mapping[node] = 10.0
        partition_mapping = {k: v for k, v in partition.items() if k in E.nodes}
        edge_weights = [(e, G.edges[e]['weight']) for e in E.edges]

        for node in E.nodes:
            E2.add_node(node,
                        label=node_names[node],
                        partition=partition_mapping[node],
                        pr=pr_mapping[node])

        for edge in edge_weights:
            E2.add_edge(edge[0][0], edge[0][1], weight=edge[1])

        pos = nx.spring_layout(E2, weight="weight", seed=42)

        data = {}
        data["nodes"] = []
        data["links"] = []
        idx = list(E.nodes)
        for node in E.nodes:
            Ego.add_node(node,
                         label=node_names[node],
                         partition=partition_mapping[node],
                         pr=pr_mapping[node],
                         pos_x=pos[node][0],
                         pos_y=pos[node][1])
            node_info = {}
            node_info["name"] = node_names[node]
            node_info["group"] = partition_mapping[node]
            data["nodes"].append(node_info)

        for edge in edge_weights:
            Ego.add_edge(edge[0][0], edge[0][1], weight=edge[1])
            source_idx = idx.index(edge[0][0])
            target_idx = idx.index(edge[0][1])
            weight = edge[1]
            edge_info = {"source": source_idx,
                         "target": target_idx,
                         "value": weight}
            data["links"].append(edge_info)
        nx.write_graphml(Ego, f"data/ego_networks/{fname}")

        # make interactive vizs
        if not os.path.isdir(f"data/ego_d3/{person_name}"):
            os.mkdir(f"data/ego_d3/{person_name}")
        else:
            rmtree(f"data/ego_d3/{person_name}")
            os.mkdir(f"data/ego_d3/{person_name}")

        with open(f"data/ego_d3/{person_name}/adjacency.json", "w") as outfile:
            json.dump(data, outfile)
        fs = [f for f in os.listdir("data/template") if os.path.isfile(
            os.path.join("data/template", f))]
        for f in fs:
            copy2(f"data/template/{f}", f"data/ego_d3/{person_name}/{f}")
