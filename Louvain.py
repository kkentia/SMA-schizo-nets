#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import networkx as nx
import numpy as np
from collections import defaultdict

file_path = "./SMA_data_processing/cobre_combined_connectomes_database.h5"

graphs_hc_p = []
graphs_scz_p = []

graphs_hc_g = []
graphs_scz_g = []

with h5py.File(file_path, "r") as f:
    hc_p = f["hc_pearson"]
    scz_p = f["scz_pearson"]

    hc_g = f["hc_glasso"]
    scz_g = f["scz_glasso"]

    for i in range(hc_p.shape[0]):
        A = hc_p[i]
        A = A.copy()
        np.fill_diagonal(A, 0)
        graphs_hc_p.append(nx.from_numpy_array(A))

    for i in range(scz_p.shape[0]):
        A = scz_p[i]
        A = A.copy()
        np.fill_diagonal(A, 0)
        graphs_scz_p.append(nx.from_numpy_array(A))

    for i in range(hc_g.shape[0]):
        A = hc_g[i]
        A = A.copy()
        np.fill_diagonal(A, 0)
        graphs_hc_g.append(nx.from_numpy_array(A))

    for i in range(scz_g.shape[0]):
        A = scz_g[i]
        A = A.copy()
        np.fill_diagonal(A, 0)
        graphs_scz_g.append(nx.from_numpy_array(A))

print("HC graphs:", len(graphs_hc_p))
print("SCZ graphs:", len(graphs_scz_p))
print("HC graphs (Glasso):", len(graphs_hc_g))
print("SCZ graphs (Glasso):", len(graphs_scz_g))
print(graphs_hc_p[0])
print(list(graphs_hc_p[0].neighbors(0)))
print(graphs_hc_g[0])
print(list(graphs_hc_g[0].neighbors(0)))

G = graphs_hc_p[0]


# Reference : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html
# 
# The algorithm works in 2 steps. On the first step it assigns every node to be in its own community and then for each node it tries to find the maximum positive modularity gain by moving each node to all of its neighbor communities. If no positive gain is achieved the node remains in its original community.
# 
# The modularity gain obtained by moving an **isolated node** into a community can easily be calculated by the following formula :
# 
# $$
# \Delta Q = \frac{k_{i,in}}{2m} - \gamma \frac{\Sigma_{tot}\cdot k_i}{2m^2}
# $$
# 
# where :
# 
# - $m$ is the size of the graph
# - $k_{i,\mathrm{in}}$ is the sum of the weights of the links from $i$ to nodes in $C$
# - $k_i$ is the sum of the weights of the links incident to node $i$
# - $\Sigma_{\mathrm{tot}}$ is the sum of the weights of the links incident to nodes in $C$
# - $\gamma$ is the resolution parameter
# 

# In[3]:


def initial_assignment_communities(G) :
    return {node: node for node in G.nodes()}


# In[4]:


def delta_q(G, node, target_community, communities, gamma=1.0, weight="weight"):

    degrees = dict(G.degree(weight=weight))
    two_m = sum(degrees.values())

    k_i_in = 0.0

    for neighbor, edge_data in G[node].items():
        if neighbor != node and communities[neighbor] == target_community:
            k_i_in += edge_data.get(weight)

    k_i = degrees[node]

    sigma_tot = 0.0

    for n in G.nodes():
        if n != node and communities[n] == target_community:
            sigma_tot += degrees[n]

    return (k_i_in / two_m) - gamma * (sigma_tot * k_i) / (two_m * two_m)


# In[5]:


def louvain_step_one(G, gamma=1.0, weight="weight"):

    communities = initial_assignment_communities(G)
    moved = True

    while moved:
        moved = False

        for node in G.nodes():
            current_community = communities[node]

            # change community to isolate node
            communities[node] = -1

            #unique values
            target_communities = set()

            for neighbor in G.neighbors(node):
                if communities[neighbor] != -1:
                    target_communities.add(communities[neighbor])

            best_community = current_community
            best_dq = 0.0

            for tc in target_communities:
                dq = delta_q(G, node, tc, communities, gamma=gamma, weight=weight)
                if dq > best_dq:
                    best_dq = dq
                    best_community = tc

            communities[node] = best_community

            if best_community != current_community:
                moved = True

    return communities


# In[6]:


communities = louvain_step_one(G)
print(communities)


# The second phase consists in building a new network whose nodes are now the communities found in the first phase. To do so, the weights of the links between the new nodes are given by the sum of the weight of the links between nodes in the corresponding two communities. Once this phase is complete it is possible to reapply the first phase creating bigger communities with increased modularity.

# In[7]:


def louvain_step_two(G, communities, weight="weight"):
    new_G = nx.Graph()

    unique_communities = set(communities.values())
    for uc in unique_communities:
        new_G.add_node(uc)

    for n1, n2, edge_data in G.edges(data=True):
        c_n1 = communities[n1]
        c_n2 = communities[n2]
        w = edge_data.get(weight)

        if new_G.has_edge(c_n1, c_n2):
            new_G[c_n1][c_n2][weight] += w
        else:
            new_G.add_edge(c_n1, c_n2, **{weight: w})

    return new_G


# In[8]:


new_G = louvain_step_two(G, communities)
print("Original :", G)
print("New : ", new_G)
print(new_G.nodes())


# The above two phases are executed until no modularity gain is achieved (or is less than the threshold, or until max_levels is reached).
# 
# $$
# Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right)\delta(c_i, c_j)
# $$

# In[ ]:


def modularity(G, communities, weight='weight'):

    degrees = dict(G.degree(weight=weight))
    two_m = sum(degrees.values())
    sigma = 0.0
    nodes = list(G.nodes())

    for i in nodes:
        for j in nodes:
            if communities[i] == communities[j]:
                if G.has_edge(i, j):
                    A_ij = G[i][j].get(weight, 1.0)
                else:
                    A_ij = 0
                sigma += A_ij - (degrees[i] * degrees[j]) / two_m

    return sigma / two_m


# In[10]:


print(modularity(G, communities))

grouped = defaultdict(set)
for node, cid in communities.items():
    grouped[cid].add(node)

communities_nx = list(grouped.values())

nx.algorithms.community.quality.modularity(G, communities_nx, weight='weight', resolution=1)

print(communities)
print(communities_nx)


# In[11]:


def louvain(G, weight="weight", threshold=1e-07, max_levels=1000, gamma=1.0):
    current_G = G
    levels = 0

    communities_original_graph = {node: node for node in G.nodes()}

    prev_modularity = -1

    while levels < max_levels:

        communities = louvain_step_one(current_G, gamma)

        curr_modularity = modularity(current_G, communities, weight=weight)

        if prev_modularity != -1 and abs(curr_modularity - prev_modularity) < threshold:
            break

        prev_modularity = curr_modularity

        new_G = louvain_step_two(current_G, communities, weight=weight)

        new_communities_original_graph = {}
        for original_node, old_comm in communities_original_graph.items():
            new_communities_original_graph[original_node] = communities[old_comm]

        communities_original_graph = new_communities_original_graph
        current_G = new_G
        levels += 1

    return communities_original_graph


# In[12]:


com_nx = nx.algorithms.community.louvain_communities(G, weight='weight', resolution=1, threshold=1e-07, max_level=1000, seed=None)
q_nx = nx.algorithms.community.quality.modularity(G, com_nx, weight='weight', resolution=1)

print(q_nx)

com = louvain(G, weight="weight", threshold=1e-07, max_levels=1000, gamma=1.0)
q = modularity(G, com)

print(q)


# In[ ]:


def delta_q(G, node, target_community, communities, degrees, two_m, comm_degrees, gamma=1.0, weight="weight"):
    k_i_in = 0.0
    for neighbor, edge_data in G[node].items():
        if neighbor != node and communities[neighbor] == target_community:
            k_i_in += edge_data.get(weight, 1.0)

    k_i = degrees[node]

    # Get sigma_tot with Dictionary lookup instead of the loop
    sigma_tot = comm_degrees.get(target_community, 0.0)

    return (k_i_in / two_m) - gamma * (sigma_tot * k_i) / (two_m * two_m)


# In[ ]:


def louvain_step_one(G, gamma=1.0, weight="weight"):
    communities = initial_assignment_communities(G)

    # ── Pre-compute once (instead of every delta_q call) ──
    degrees = dict(G.degree(weight=weight))
    two_m = sum(degrees.values())

    # Build comm_degrees: community_id → sum of degrees of its members
    comm_degrees = {}
    for node, comm in communities.items():
        comm_degrees[comm] = comm_degrees.get(comm, 0.0) + degrees[node]

    moved = True

    while moved:
        moved = False

        for node in G.nodes():
            current_community = communities[node]
            k_i = degrees[node]

            # Remove node from its current community
            communities[node] = -1
            comm_degrees[current_community] -= k_i

            # Find neighbor communities
            target_communities = set()
            for neighbor in G.neighbors(node):
                if communities[neighbor] != -1:
                    target_communities.add(communities[neighbor])
            best_community = current_community
            best_dq = 0.0

            for tc in target_communities:
                dq = delta_q(G, node, tc, communities, degrees, two_m, comm_degrees, gamma=gamma, weight=weight)
                if dq > best_dq:
                    best_dq = dq
                    best_community = tc

            # Place node in best community and update comm_degrees
            communities[node] = best_community
            comm_degrees[best_community] = comm_degrees.get(best_community, 0.0) + k_i

            if best_community != current_community:
                moved = True

    return communities


# In[18]:


q_hc_p_tot = 0
q_scz_p_tot = 0

n_hc_p = len(graphs_hc_p)
n_scz_p = len(graphs_scz_p)

for graph in graphs_hc_p :
    com = louvain(graph, weight="weight", threshold=1e-07, max_levels=1000, gamma=1.0)
    q = modularity(graph, com)
    q_hc_p_tot += q

for graph in graphs_scz_p :
    com = louvain(graph, weight="weight", threshold=1e-07, max_levels=1000, gamma=1.0)
    q = modularity(graph, com)
    q_scz_p_tot += q

print(f"Healthy controls, Pearson: {q_hc_p_tot/n_hc_p}")
print(f"Schizophrenia, Pearson: {q_scz_p_tot/n_scz_p}")

q_hc_g_tot = 0
q_scz_g_tot = 0

n_hc_g = len(graphs_hc_g)
n_scz_g = len(graphs_scz_g)

for graph in graphs_hc_g :
    com = louvain(graph, weight="weight", threshold=1e-07, max_levels=1000, gamma=1.0)
    q = modularity(graph, com)
    q_hc_g_tot += q

for graph in graphs_scz_g :
    com = louvain(graph, weight="weight", threshold=1e-07, max_levels=1000, gamma=1.0)
    q = modularity(graph, com)
    q_scz_g_tot += q

print(f"Healthy controls, Glasso: {q_hc_g_tot/n_hc_g}")
print(f"Schizophrenia, Glasso: {q_scz_g_tot/n_scz_g}")


# In[19]:


q_hc_p_tot_nx = 0
q_scz_p_tot_nx = 0

n_hc_p_nx = len(graphs_hc_p)
n_scz_p_nx = len(graphs_scz_p)

for graph in graphs_hc_p :
    com_nx = nx.algorithms.community.louvain_communities(graph, weight='weight', resolution=1, threshold=1e-07, max_level=1000, seed=None)
    q_nx = nx.algorithms.community.quality.modularity(graph, com_nx, weight='weight', resolution=1)
    q_hc_p_tot_nx += q_nx

for graph in graphs_scz_p :
    com_nx = nx.algorithms.community.louvain_communities(graph, weight='weight', resolution=1, threshold=1e-07, max_level=1000, seed=None)
    q_nx = nx.algorithms.community.quality.modularity(graph, com_nx, weight='weight', resolution=1)
    q_scz_p_tot_nx += q_nx

print(f"Healthy Controls, Pearson: {q_hc_p_tot_nx}/{n_hc_p_nx}")
print(f"Schizophrenia, Pearson: {q_scz_p_tot_nx}/{n_scz_p_nx}")

q_hc_g_tot_nx = 0
q_scz_g_tot_nx = 0

n_hc_g_nx = len(graphs_hc_g)
n_scz_g_nx = len(graphs_scz_g)

for graph in graphs_hc_g :
    com_nx = nx.algorithms.community.louvain_communities(graph, weight='weight', resolution=1, threshold=1e-07, max_level=1000, seed=None)
    q_nx = nx.algorithms.community.quality.modularity(graph, com_nx, weight='weight', resolution=1)
    q_hc_g_tot_nx += q_nx

for graph in graphs_scz_g :
    com_nx = nx.algorithms.community.louvain_communities(graph, weight='weight', resolution=1, threshold=1e-07, max_level=1000, seed=None)
    q_nx = nx.algorithms.community.quality.modularity(graph, com_nx, weight='weight', resolution=1)
    q_scz_g_tot_nx += q_nx

print(f"Healthy Controls, Glasso: {q_hc_g_tot_nx}/{n_hc_g_nx}")
print(f"Schizophrenia, Glasso: {q_scz_g_tot_nx}/{n_scz_g_nx}")


# In[ ]:




