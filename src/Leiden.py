#!/usr/bin/env python
# coding: utf-8

# # Leiden Algorithm
# Custom implementation following **Traag, Waltman & van Eck (2019)**,
# _"From Louvain to Leiden: guaranteeing well-connected communities"_.
# 
# All line references (e.g. "Line 14") refer to **Algorithm A.2** in the paper's
# Supplementary Information.

# In[ ]:


import networkx as nx
import numpy as np
import random
from collections import deque, defaultdict


# In[ ]:


def delta_q(G, node, target_community, communities, degrees, two_m, comm_degrees, gamma=1.0, weight="weight"):
    """
    ΔQ for moving `node` into `target_community`.

    ΔQ = k_{v→C}/2m − γ·σ_C·k_v/(2m)²

    - degrees, two_m, comm_degrees are pre-computed by the caller.
    - O(degree(v)) per call thanks to the comm_degrees dict lookup.
    """
    k_i_in = 0.0
    for neighbor, edge_data in G[node].items():
        if neighbor != node and communities.get(neighbor) == target_community:
            k_i_in += edge_data.get(weight, 1.0)
    k_i = degrees[node]
    sigma_tot = comm_degrees.get(target_community, 0.0)
    return (k_i_in / two_m) - gamma * (sigma_tot * k_i) / (two_m * two_m)


# In[ ]:


def singleton_partition(G):
    """Each node starts as its own community. Used at the start of each iteration."""
    return {node: node for node in G.nodes()}


# In[ ]:


def aggregate_graph(G, partition, weight="weight"):
    """
    Collapse G so each community becomes one super-node.
    Inter-community edges are summed; intra-community edges become self-loops.
    Preserves total weight 2m across iterations.
    """
    new_G = nx.Graph()
    for comm in set(partition.values()):
        new_G.add_node(comm)
    for u, v, data in G.edges(data=True):
        comm_u, comm_v = partition[u], partition[v]
        w = data.get(weight, 1.0)
        if new_G.has_edge(comm_u, comm_v):
            new_G[comm_u][comm_v][weight] += w
        else:
            new_G.add_edge(comm_u, comm_v, **{weight: w})
    return new_G


# In[ ]:


def move_nodes_fast(G, partition, weight="weight", gamma=1.0):
    """
    MoveNodesFast — Lines 13-24.

    Line 14: Init queue Q with all nodes in random order.
    Line 16: Pop next node v from Q.
    Line 17: Find C' = argmax ΔQ among neighbouring communities.
    Line 18: Move only if ΔQ > 0.
    Lines 20-21: Re-enqueue neighbours of v not already in Q.
    Line 23: Repeat until Q is empty.
    """
    nodes = list(G.nodes())
    random.shuffle(nodes)
    queue = deque(nodes)                        # Line 14
    in_queue = {n: True for n in nodes}

    degrees = dict(G.degree(weight=weight))
    two_m = sum(degrees.values())
    comm_degrees = defaultdict(float)
    for node, comm in partition.items():
        comm_degrees[comm] += degrees[node]

    while queue:                                # Line 23
        v = queue.popleft()                     # Line 16
        in_queue[v] = False
        current_comm = partition[v]
        comm_degrees[current_comm] -= degrees[v]

        neighbor_comms = set(partition[nb] for nb in G.neighbors(v))
        neighbor_comms.add(current_comm)

        best_comm = current_comm                # Line 17
        max_dq = 0.0
        for comm in neighbor_comms:
            dq = delta_q(G=G, node=v, target_community=comm, communities=partition,
                         degrees=degrees, two_m=two_m, comm_degrees=comm_degrees,
                         gamma=gamma, weight=weight)
            if dq > max_dq:
                max_dq = dq
                best_comm = comm

        if max_dq > 0 and best_comm != current_comm:   # Line 18
            partition[v] = best_comm                    # Line 19
            for neighbor in G.neighbors(v):             # Lines 20-21
                if not in_queue[neighbor]:
                    queue.append(neighbor)
                    in_queue[neighbor] = True

        comm_degrees[best_comm] += degrees[v]

    return partition


# In[ ]:


def merge_nodes_subset(G, P_refined, S, theta=0.01, gamma=1.0, weight="weight"):
    """
    MergeNodesSubset — Lines 33-42.

    Line 34: Gate — node v must be well-connected to S\\{v}.
    Line 36: Gate — v must still be in a singleton community in P_refined.
    Line 37: Gate — candidate community C must be well-connected to S\\C.
    Line 38: Sample destination randomly ∝ exp(ΔQ/θ) among candidates with ΔQ ≥ 0.
    Line 39: Move v to chosen community.
    """
    S_set = set(S)
    degrees = dict(G.degree(weight=weight))
    two_m = sum(degrees.values())

    comm_degrees = defaultdict(float)
    comm_sizes = defaultdict(int)
    for node in S:
        comm = P_refined[node]
        comm_degrees[comm] += degrees[node]
        comm_sizes[comm] += 1

    K_S = sum(degrees[u] for u in S)

    R = list(S)
    random.shuffle(R)

    for v in R:
        # Line 36: singleton gate
        if comm_sizes[P_refined[v]] > 1:
            continue

        # Line 34: node well-connectedness gate
        k_v_S = sum(data.get(weight, 1.0) for nb, data in G[v].items()
                     if nb in S_set and nb != v)
        if k_v_S / two_m < gamma * (degrees[v] * (K_S - degrees[v])) / (two_m * two_m):
            continue

        # Candidate communities from v's neighbours within S
        T = set(P_refined[nb] for nb in G.neighbors(v)
                if nb in S_set and P_refined[nb] != P_refined[v])

        # "Stay" option: ΔQ = 0 → weight = exp(0) = 1.0
        candidate_comms = [P_refined[v]]
        candidate_probs = [1.0]

        for comm in T:
            # Line 37: community well-connectedness gate
            ext_w = 0.0
            for u in S:
                if P_refined[u] == comm:
                    for nb, data in G[u].items():
                        if nb in S_set and P_refined[nb] != comm:
                            ext_w += data.get(weight, 1.0)
            c_deg = comm_degrees.get(comm, 0.0)
            if ext_w / two_m < gamma * (c_deg * (K_S - c_deg)) / (two_m * two_m):
                continue

            # Line 38: Boltzmann-weighted random sampling
            dq = delta_q(G=G, node=v, target_community=comm, communities=P_refined,
                         degrees=degrees, two_m=two_m, comm_degrees=comm_degrees,
                         gamma=gamma, weight=weight)
            if dq >= 0:
                prob = np.exp((1 / theta) * dq)
                candidate_comms.append(comm)
                candidate_probs.append(prob)

        # Line 38: sample destination
        if len(candidate_comms) > 1:
            best_comm = random.choices(candidate_comms, weights=candidate_probs, k=1)[0]
            if best_comm != P_refined[v]:       # Line 39
                old_comm = P_refined[v]
                comm_degrees[old_comm] -= degrees[v]
                comm_sizes[old_comm] -= 1
                P_refined[v] = best_comm
                comm_degrees[best_comm] += degrees[v]
                comm_sizes[best_comm] += 1

    return P_refined


# In[ ]:


def refine_partition(G, P, theta=0.01, gamma=1.0, weight="weight"):
    """
    RefinePartition — Lines 26-31.

    Line 27: Start from a singleton partition P_refined.
    Lines 28-29: For each community C in P, merge singletons within C
                 via merge_nodes_subset. P_refined is always a refinement of P.
    """
    P_refined = singleton_partition(G)          # Line 27

    communities_in_P = defaultdict(set)
    for node, comm in P.items():
        communities_in_P[comm].add(node)

    for comm, S in communities_in_P.items():    # Lines 28-29
        P_refined = merge_nodes_subset(G, P_refined, S, theta, gamma, weight)

    return P_refined


# In[ ]:

def leiden_algorithm(G, theta=0.01, gamma=1.0, weight="weight", max_iterations=50):
    current_G = G
    P = singleton_partition(current_G)
    node_mapping = {n: n for n in G.nodes()}
    done = False

    # 1. Add an iteration counter
    iterations = 0

    while not done and iterations < max_iterations:
        iterations += 1

        P = move_nodes_fast(current_G, P, weight, gamma)
        if len(set(P.values())) == current_G.number_of_nodes():
            done = True

        if not done:
            P_refined = refine_partition(current_G, P, theta, gamma, weight)
            new_G = aggregate_graph(current_G, P_refined, weight)

            # 2. Prevent infinite loops if the graph stops shrinking
            if new_G.number_of_nodes() == current_G.number_of_nodes():
                break

            new_node_mapping = {}
            for original_node, current_node in node_mapping.items():
                new_node_mapping[original_node] = P_refined.get(current_node, current_node)
            node_mapping = new_node_mapping

            current_G = new_G
            P = singleton_partition(current_G)

    return node_mapping



def leiden_algorithm_old(G, theta=0.01, gamma=1.0, weight="weight"):
    """
    Leiden — Lines 1-11.

    Line 3:  MoveNodesFast to locally optimise Q.
    Line 4:  Stop when P is all singletons (no moves were made).
    Line 6:  RefinePartition — may split communities from Phase 1.
    Line 7:  Aggregate graph on P_refined.
    Line 8:  Re-init P from the non-refined partition.
    Line 11: Return flat partition mapped to original node labels.
    """
    current_G = G
    P = singleton_partition(current_G)
    node_mapping = {n: n for n in G.nodes()}
    done = False

    while not done:
        P = move_nodes_fast(current_G, P, weight, gamma)    # Line 3
        if len(set(P.values())) == current_G.number_of_nodes():  # Line 4
            done = True

        if not done:
            P_refined = refine_partition(current_G, P, theta, gamma, weight)   # Line 6
            new_G = aggregate_graph(current_G, P_refined, weight)               # Line 7

            # Line 8: map original nodes through P_refined
            new_node_mapping = {}
            for original_node, current_node in node_mapping.items():
                new_node_mapping[original_node] = P_refined.get(current_node, current_node)
            node_mapping = new_node_mapping

            current_G = new_G
            P = singleton_partition(current_G)

    return node_mapping                                     # Line 11


# In[ ]:


def modularity_vectorized(G, communities, weight='weight'):
    """
    Q = (1/2m) · Σ_ij [A_ij − k_i·k_j/2m] · δ(c_i, c_j)

    Vectorized with NumPy for ~100-1000× speedup over a Python double-loop.
    """
    nodes = list(G.nodes())
    if not nodes:
        return 0.0
    A = nx.to_numpy_array(G, nodelist=nodes, weight=weight)
    k = np.array([G.degree(n, weight=weight) for n in nodes])
    two_m = k.sum()
    if two_m == 0:
        return 0.0
    expected = np.outer(k, k) / two_m
    B = A - expected
    c_array = np.array([communities.get(n, -1) for n in nodes])
    delta_matrix = (c_array[:, None] == c_array).astype(int)
    return np.sum(B * delta_matrix) / two_m


# In[ ]:


def louvain_move_nodes(G, partition, weight="weight", gamma=1.0):
    """
    Louvain local-move phase: greedily move each node to the best
    neighbouring community. Sweep all nodes, repeat until stable.
    """
    degrees = dict(G.degree(weight=weight))
    two_m = sum(degrees.values())
    comm_degrees = defaultdict(float)
    for node, comm in partition.items():
        comm_degrees[comm] += degrees[node]

    moved = True
    while moved:
        moved = False
        for node in G.nodes():
            current_comm = partition[node]
            comm_degrees[current_comm] -= degrees[node]

            neighbor_comms = set(partition[nb] for nb in G.neighbors(node))
            neighbor_comms.add(current_comm)

            best_comm = current_comm
            best_dq = 0.0
            for comm in neighbor_comms:
                dq = delta_q(G=G, node=node, target_community=comm, communities=partition,
                             degrees=degrees, two_m=two_m, comm_degrees=comm_degrees,
                             gamma=gamma, weight=weight)
                if dq > best_dq:
                    best_dq = dq
                    best_comm = comm

            partition[node] = best_comm
            comm_degrees[best_comm] += degrees[node]

            if best_comm != current_comm:
                moved = True

    return partition


def louvain_algorithm(G, weight="weight", gamma=1.0, threshold=1e-7, max_levels=1000):
    """
    Full Louvain: alternates local-move + aggregation until Q stabilises.
    """
    current_G = G
    node_mapping = {n: n for n in G.nodes()}
    prev_q = -1.0

    for _ in range(max_levels):
        partition = singleton_partition(current_G)
        partition = louvain_move_nodes(current_G, partition, weight=weight, gamma=gamma)

        curr_q = modularity_vectorized(current_G, partition, weight=weight)
        if abs(curr_q - prev_q) < threshold:
            break
        prev_q = curr_q

        new_G = aggregate_graph(current_G, partition, weight=weight)

        new_node_mapping = {}
        for orig, curr in node_mapping.items():
            new_node_mapping[orig] = partition[curr]
        node_mapping = new_node_mapping

        current_G = new_G

    return node_mapping

