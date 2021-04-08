import networkx as nx
import numpy as np
from sklearn_extra.cluster import KMedoids
import copy
import math


def find_attack_set_by_degree(adj, b, B):
    G = nx.from_numpy_matrix(adj)
    D = G.degree()
    Degree = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        Degree[i] = D[i]
    # print(Degree)
    Dsort = Degree.argsort()[::-1]
    l = Dsort
    chosen_nodes = []
    i = 0
    k = 0
    while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
        while calculate_total_budget_of_attack_set(b, add_node([l[i]], chosen_nodes)) > B:
            i = i + 1
            if i == adj.shape[0]:
                k = 1
                break
        else:
            chosen_nodes.append(l[i])
            i = i + 1
            if i == adj.shape[0]:
                break
        if k == 1:
            break
    return np.array(chosen_nodes)



def add_node(set1, set2):
    intersection = list(set(set1).union(set(set2)))
    return np.array(intersection)


def find_neighbor(adj, attack_set, k):
    G = nx.from_numpy_matrix(adj)
    attack_num = attack_set.shape[0]
    target_set = []
    for i in range(attack_num):
        set1 = np.array([n for n in G.neighbors(attack_set[i])])
        target_set = add_node(target_set, set1)
    # print(set)

    return np.array(list(set(target_set).difference(set(attack_set))))


def calculate_goal(yp, pred):
    "optimization goal: calculate the average difference between original prediction and prediction under attack"
    difference = np.zeros(yp.shape[1])
    for i in range(yp.shape[1]):
        difference[i] = yp[0, i] - pred[0, i]
    return np.mean(difference)


def find_attack_set_by_kmedoids(locations, b, B):
    at_num = 1
    chosen_nodes = []
    temp = []
    while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
        chosen_nodes = []
        kmedoids = KMedoids(n_clusters=at_num, random_state=0).fit(locations)
        centers = kmedoids.cluster_centers_
        for i in range(at_num):
            for j in range(locations.shape[0]):
                if locations[j, 0] == centers[i, 0] and locations[j, 1] == centers[i, 1]:
                    chosen_nodes.append(j)
        if calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
            temp = copy.deepcopy(chosen_nodes)  # record the chosen nodes in last iteration
            at_num = at_num + 1
            if at_num == b.shape[0]:
                break
        else:
            break
    return np.array(temp)


def find_attack_set_by_pagerank(adj, b, B):
    G = nx.from_numpy_matrix(adj)
    result = nx.pagerank(G)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]  # The sequence produced by pagerank algorithm
    chosen_nodes = []
    i = 0
    k = 0
    while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
        while calculate_total_budget_of_attack_set(b, add_node([l[i]], chosen_nodes)) > B:
            i = i + 1
            if i == adj.shape[0]:
                k = 1
                break
        else:
            chosen_nodes.append(l[i])
            i = i + 1
            if i == adj.shape[0]:
                break
        if k == 1:
            break
    return np.array(chosen_nodes)


def find_attack_set_by_Kg_pagerank(adj, b, B):
    G = nx.from_numpy_matrix(adj)
    result = nx.pagerank(G)
    for i in range(adj.shape[0]):
        result[i] = result[i] / b[i]
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]  # The sequence produced by pagerank algorithm
    chosen_nodes = []
    i = 0
    k = 0
    while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
        while calculate_total_budget_of_attack_set(b, add_node([l[i]], chosen_nodes)) > B:
            i = i + 1
            if i == adj.shape[0]:
                k = 1
                break
        else:
            chosen_nodes.append(l[i])
            i = i + 1
            if i == adj.shape[0]:
                break
        if k == 1:
            break
    return np.array(chosen_nodes)


def find_attack_set_by_betweenness(adj, b, B):
    G = nx.from_numpy_matrix(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = []
    i = 0
    k = 0
    while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
        while calculate_total_budget_of_attack_set(b, add_node([l[i]], chosen_nodes)) > B:
            i = i + 1
            if i == adj.shape[0]:
                k = 1
                break
        else:
            chosen_nodes.append(l[i])
            i = i + 1
            if i == adj.shape[0]:
                break
        if k == 1:
            break
    return np.array(chosen_nodes)


def find_attack_set_by_Kg_betweenness(adj, b, B):
    G = nx.from_numpy_matrix(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    for i in range(adj.shape[0]):
        result[i] = result[i] / b[i]
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = []
    i = 0
    k = 0
    while calculate_total_budget_of_attack_set(b, chosen_nodes) < B:
        while calculate_total_budget_of_attack_set(b, add_node([l[i]], chosen_nodes)) > B:
            i = i + 1
            if i == adj.shape[0]:
                k = 1
                break
        else:
            chosen_nodes.append(l[i])
            i = i + 1
            if i == adj.shape[0]:
                break
        if k == 1:
            break
    return np.array(chosen_nodes)


def find_k_hop_neighbor_of_single_node(adj, source, k):
    if k == 0:
        return k
    elif k > 0:
        G = nx.from_numpy_matrix(adj)
        k_hop_neighbor1 = nx.single_source_shortest_path_length(G, source, cutoff=k)
        k_hop_neighbor2 = nx.single_source_shortest_path_length(G, source, cutoff=k-1)
        k_hop_neighbor = k_hop_neighbor1.keys() - k_hop_neighbor2.keys()
        return np.array(list(k_hop_neighbor))


def calculate_degree_of_nodes_in_set(adj, nodes_set):
    nodes_set = np.array(nodes_set)
    corresponding_degree = np.zeros(nodes_set.shape)
    G = nx.from_numpy_matrix(adj)
    D = np.array(G.degree())
    # print(D)
    for i in range(len(nodes_set)):
        corresponding_degree[i] = D[nodes_set[i], 1]
    return corresponding_degree


def calculate_total_budget_of_attack_set(b, attack_set):
    budget = 0
    if attack_set is None:  # empty list
        return 0
    else:
        for i in range(len(attack_set)):
            budget = budget + b[attack_set[i]]
        return budget
