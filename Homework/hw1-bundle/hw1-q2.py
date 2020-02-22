import snap
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

g_coauthor = None
TARGET_ID = 9

def load_data(file_name: str) -> snap.TUNGraph:
    return snap.TUNGraph.Load(snap.TFIn(file_name))


def node_features(node: snap.TUNGraphNodeI, graph: snap.TUNGraph) -> np.array:
    node_v = snap.TIntV()
    node_v.Add(node.GetId())

    degree = node.GetDeg()

    for i in range(degree):
        node_v.Add(node.GetNbrNId(i))

    number_of_edges_in_egonet, number_of_edges_connect_rest = snap.GetEdgesInOut(graph, node_v)

    return np.array([degree, number_of_edges_in_egonet, number_of_edges_connect_rest])


def consine_similarity(v1: np.array, v2: np.array) -> float:
    num = 0.0
    denom = 0.0

    num = np.sum(v1 * v2)

    denom = np.sqrt(np.sum(v1 * v1) * np.sum(v2 * v2))

    if abs(denom - 0.0) < 1e-6:
        return 0.0

    return num / denom


def Q2_1():
    global g_coauthor
    g_coauthor = load_data("hw1-q2.graph")

    nid_v = snap.TIntV()
    g_coauthor.GetNIdV(nid_v)
    sim_dict = {}

    target_node = g_coauthor.GetNI(TARGET_ID)
    target_v = node_features(target_node, g_coauthor)
    if target_v[0] > 10 or target_v[1] > 10 or target_v[2] > 10:
        raise BaseException("Internal error.")

    for item in nid_v:
        if item != TARGET_ID:
            n = g_coauthor.GetNI(item)
            v = node_features(n, g_coauthor)
            sim = consine_similarity(target_v, v)

            sim_dict[item] = sim

    sorted_lst = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True) # return a list
    print(f"The top 5 nodes that are most similar to node {TARGET_ID}:")
    for i in range(5):
        print(sorted_lst[i])


Q2_1()


def generate_features(node: snap.TUNGraphNodeI, graph: snap.TUNGraph, n: int) -> np.array:
    if n == 0:
        return node_features(node, graph)

    a1 = generate_features(node, graph, n - 1)
    feature_sum = np.zeros_like(a1, dtype=np.float)
    d = node.GetDeg()
    if d == 0:
        return np.concatenate((a1, np.zeros_like(a1), np.zeros_like(a1)))

    for i in range(d):
        node_id = node.GetNbrNId(i)
        nbr_n = graph.GetNI(node_id)
        feature_sum += generate_features(nbr_n, graph, n - 1)

    a2 = feature_sum / d

    return np.concatenate((a1, a2, feature_sum))


def calculate_similarity(graph: snap.TUNGraph, n: int) -> dict:
    nid_v = snap.TIntV()
    graph.GetNIdV(nid_v)
    sim_dict = {}

    target_node = graph.GetNI(TARGET_ID)
    target_v = generate_features(target_node, graph, n)
    # print(target_v.shape)

    for item in nid_v:
        if item != TARGET_ID:
            n_n = graph.GetNI(item)
            v = generate_features(n_n, graph, n)
            sim = consine_similarity(target_v, v)

            sim_dict[item] = sim
    
    return sim_dict


def Q2_2():
    sim_dict = calculate_similarity(g_coauthor, 2)
    sorted_lst = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True)

    print(f"The top 5 nodes that are most similar to node {TARGET_ID}:")
    for i in range(5):
        print(sorted_lst[i])


Q2_2()


def draw_histogram(sim_dict: dict):
    sim_v = list(sim_dict.values())
    # x, y = np.histogram(res, 20)
    plt.hist(sim_v, 20)
    plt.title(f"Similarity histogram of node {TARGET_ID}")
    plt.show()


def choose_one_node_in_range(sim_dict: dict, l: float, r: float):
    candidates = []
    for k, v in sim_dict.items():
        if v >= l and v < r:
            if g_coauthor.GetNI(k).GetDeg() > 3:
                candidates.append(k)
    
    return candidates[np.random.randint(0, len(candidates))]


def draw_subgraph(graph: snap.TUNGraph, node_id: int, with_edge_labels=False) -> list:
    g = nx.Graph()
    g.add_node(node_id)
    node_iter = graph.GetNI(node_id)
    d = node_iter.GetDeg()
    nbr_v = [node_id]
    
    for i in range(d):
        nbr_id = node_iter.GetNbrNId(i)
        g.add_node(nbr_id)
        nbr_v.append(nbr_id)

    for i in nbr_v:
        for j in nbr_v:
            if i != j and graph.IsEdge(i, j):
                f_i = generate_features(graph.GetNI(i), graph, 2)
                f_j = generate_features(graph.GetNI(j), graph, 2)
                g.add_edge(i, j, weight=round(consine_similarity(f_i, f_j), 4))

    nx.draw(g, with_labels=True)
    if with_edge_labels:
        labels = nx.get_edge_attributes(g, 'weight')
        pos = nx.spring_layout(g)
        nx.draw_networkx_edge_labels(g, pos, with_labels=True, edge_labels=labels)
    
    print(f"subgraph for node {node_id}")
    plt.show()

    return nbr_v


def get_neighbor_id(node_iter: snap.TUNGraphNodeI, graph: snap.TUNGraph) -> [int]:
    d = node_iter.GetDeg()
    nbr_v = []
    
    for i in range(d):
        nbr_v.append(node_iter.GetNbrNId(i))
    
    return nbr_v


def Q2_3():
    sim_dict = calculate_similarity(g_coauthor, 2)
    draw_histogram(sim_dict)

    l1, r1 = 0.60, 0.65
    l2, r2 = 0.85, 0.95
    n1 = choose_one_node_in_range(sim_dict, l1, r1)
    n2 = choose_one_node_in_range(sim_dict, l2, r2)
    draw_subgraph(g_coauthor, n1, True)
    draw_subgraph(g_coauthor, n2, True)


Q2_3()