import networkx as nx
from scipy.stats import percentileofscore
import numpy as np
import pandas as pd


def disparity_integral(x, k):
    """
    calculate the definite integral for the PDF in the disparity filter
    """
    assert x != 1.0, "x == 1.0"
    assert k != 1.0, "k == 1.0"
    return ((1.0 - x) ** k) / ((k - 1.0) * (x - 1.0))


def get_disparity_significance(norm_weight, degree):
    """
    calculate the significance (alpha) for the disparity filter
    """
    return 1.0 - (
        (degree - 1.0)
        * (disparity_integral(norm_weight, degree) - disparity_integral(0.0, degree))
    )


def disparity_filter(graph):
    """
    implements a disparity filter, based on multiscale backbone networks
    https://arxiv.org/pdf/0904.2389.pdf
    """
    alpha_measures = []

    for node_id in graph.nodes():
        node = graph.nodes[node_id]
        degree = graph.degree(node_id)
        strength = 0.00000000000000001

        for id0, id1 in graph.edges(nbunch=[node_id]):
            edge = graph[id0][id1]
            strength += edge["weight"]

        node["strength"] = strength

        for id0, id1 in graph.edges(nbunch=[node_id]):
            edge = graph[id0][id1]

            norm_weight = edge["weight"] / strength
            edge["norm_weight"] = norm_weight

            if degree > 1:
                try:
                    if norm_weight == 1.0:
                        norm_weight -= 0.0001

                    alpha = get_disparity_significance(norm_weight, degree)
                except AssertionError:
                    report_error("disparity {}".format(repr(node)), fatal=True)

                edge["alpha"] = alpha
                alpha_measures.append(alpha)
            else:
                edge["alpha"] = 0.0

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]
        edge["alpha_ptile"] = percentileofscore(alpha_measures, edge["alpha"]) / 100.0

    return alpha_measures


def calc_centrality(graph, min_degree=1):
    """
    to conserve compute costs, ignore centrality for nodes below `min_degree`
    """
    sub_graph = graph.copy()
    sub_graph.remove_nodes_from([n for n, d in list(graph.degree) if d < min_degree])

    centrality = nx.betweenness_centrality(sub_graph, weight="weight")
    # centrality = nx.closeness_centrality(sub_graph, distance="distance")

    return centrality


def calc_quantiles(metrics, num):
    """
    calculate `num` quantiles for the given list
    """
    # global DEBUG

    bins = np.linspace(0, 1, num=num, endpoint=True)
    s = pd.Series(metrics)
    q = s.quantile(bins, interpolation="nearest")

    # try:
    #     dig = np.digitize(metrics, q) - 1
    # except ValueError as e:
    #     print("ValueError:", str(e), metrics, s, q, bins)
    #     sys.exit(-1)

    quantiles = []

    for idx, q_hi in q.iteritems():
        quantiles.append(q_hi)

        # if DEBUG:
        #     print(idx, q_hi)

    return quantiles


def calc_alpha_ptile(alpha_measures, show=True):
    """
    calculate the quantiles used to define a threshold alpha cutoff
    """
    quantiles = calc_quantiles(alpha_measures, num=10)
    num_quant = len(quantiles)

    if show:
        print("\tptile\talpha")

        for i in range(num_quant):
            percentile = i / float(num_quant)
            print("\t{:0.2f}\t{:0.4f}".format(percentile, quantiles[i]))

    return quantiles, num_quant


def cut_graph(graph, min_alpha_ptile=0.5, min_degree=2):
    """
    apply the disparity filter to cut the given graph
    """
    filtered_set = set([])

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]

        if edge["alpha_ptile"] < min_alpha_ptile:
            filtered_set.add((id0, id1))

    for id0, id1 in filtered_set:
        graph.remove_edge(id0, id1)

    filtered_set = set([])

    for node_id in graph.nodes():
        node = graph.nodes[node_id]

        if graph.degree(node_id) < min_degree:
            filtered_set.add(node_id)

    for node_id in filtered_set:
        graph.remove_node(node_id)
