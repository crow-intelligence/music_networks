import numpy as np
import src.graph_viz.net3d as n3

pts = []
pts_weights = []
edg = []
edg_weights = []
with open("data/node_coords.tsv", "r") as f:
    for l in f:
        l = l.strip()
        x, y, z, w = l.split("\t")
        p = [x, y, z]
        p = [float(e) for e in p]
        w = float(w)
        pts.append(p)
        pts_weights.append(w)

with open("data/edge_coords.tsv", "r") as f:
    for l in f:
        l = l.strip().split("\t")
        x1, y1, z1, x2, y2, z2, we = l
        we = float(we)
        p1 = [x1, y1, z1]
        p2 = [x2, y2, z2]
        p1 = [float(e) for e in p1]
        p2 = [float(e) for e in p2]
        # if p1 not in pts:
        #     pts.append(p1)
        #     pts_weights.append(w1)
        # if p2 not in pts:
        #     pts.append(p2)
        #     pts_weights.append(w2)
        n1 = pts.index(p1)
        n2 = pts.index(p2)
        e = sorted([n1, n2])
        e = [e[0], e[1], we]
        if e not in edg:
            edg.append(e)
            edg_weights.append(we)

pts = np.array(pts)

rn = 1
th = 1
k, A = 50.0, 3e2
params = {
    "links": {
        "k": k,  # spring constant for links (increase if links don't contract well (not becoming straight enough), decrease if links cross a lot)
        "amplitude": A,  # strength of link repulsion (if many crossings ==> increase, decrease if layout explodes )
        "thickness": edg_weights,  # thickness of links
        "Temp0": 1,
        "ce": 300,  # noise parameters, if too noisy, set 'Temp0':0
        "segs": 5,  # # segments along link, increase if too many crossing, decrease if simulation is slow or memory error
        "weighted": True,
    },
    "nodes": {
        "amplitude": A,  # 500, # repulsion for nodes
        "radius": pts_weights,  # node size (only used if 'networkBase(...,fixed=False,...)' )
        "weighted": True,  # weighted by degree
        #                'labels': node_labels,
    },
}


nn = n3.networkBase(
    pts=pts,
    edg=edg,
    max_workers=50,
    fixed=False,  # fixed: (True: nodes are fixed), (false: nodes can move)
    **params
)


c0 = 0.1  # 0.3
tol = 0.02  # c0/5
n3.iter_converge(
    nn,
    save_its=False,  # True: save intermediate steps
    save_path="data/jss",  # where to save
    its=20,  # # of iterations before plotting and checking convergence conditions
    max_its=10500,  # maximum iterations
    c0=c0,  # 0.5, decrease if too many fluctuations in plot maybe to c0=.1 , increse if too slow, but not more than c0=1
    tol=tol,  # threshold for convergence check. increase if it doesn't converge soon enough, but not more than c0
    ellipsoid=0,
)
