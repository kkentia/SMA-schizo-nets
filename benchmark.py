#!/usr/bin/env python
# coding: utf-8

# # Community Detection Benchmark
# Compares **Custom Leiden**, **Custom Louvain**, and **NetworkX Louvain**
# on COBRE connectome data (Pearson & Glasso matrices).
# 
# Memory is cleared between matrix types to prevent RAM exhaustion.

# In[1]:


import h5py
import networkx as nx
import numpy as np
import gc
from scipy import stats
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from Leiden import (
    leiden_algorithm,
    louvain_algorithm,
    modularity_vectorized,
)


# In[2]:


def run_nx_louvain(graph):
    if graph.size(weight='weight') == 0:
        return None
    com = nx.algorithms.community.louvain_communities(graph, weight='weight')
    return nx.algorithms.community.quality.modularity(graph, com, weight='weight')


def run_custom_louvain(graph):
    if graph.size(weight='weight') == 0:
        return None
    com = louvain_algorithm(graph, weight="weight", gamma=1.0)
    return modularity_vectorized(graph, com, weight="weight")


def run_custom_leiden(graph):
    if graph.size(weight='weight') == 0:
        return None
    com = leiden_algorithm(graph, theta=0.01, gamma=1.0, weight="weight")
    return modularity_vectorized(graph, com, weight="weight")


def run_and_report(label, hc_graphs, scz_graphs, run_fn):
    hc_raw  = Parallel(n_jobs=-1)(delayed(run_fn)(g) for g in tqdm(hc_graphs,  desc=f"{label} HC"))
    scz_raw = Parallel(n_jobs=-1)(delayed(run_fn)(g) for g in tqdm(scz_graphs, desc=f"{label} SCZ"))
    hc_scores  = [s for s in hc_raw  if s is not None]
    scz_scores = [s for s in scz_raw if s is not None]
    _, p_val = stats.ttest_ind(hc_scores, scz_scores, equal_var=False)
    print(f"  Avg HC  Q-Score: {np.mean(hc_scores):.4f}")
    print(f"  Avg SCZ Q-Score: {np.mean(scz_scores):.4f}")
    print(f"  P-value (Welch): {p_val:.4e}")

def load_graphs(file_path, hc_key, scz_key, threshold=0.3):
    hc_graphs, scz_graphs = [], []
    with h5py.File(file_path, "r") as f:
        # Load HC
        for i in range(f[hc_key].shape[0]):
            A = f[hc_key][i].copy()
            np.fill_diagonal(A, 0)
            
            # Apply threshold (keeping strong positive AND negative correlations)
            A[np.abs(A) < threshold] = 0.0
            
            hc_graphs.append(nx.from_numpy_array(A))
            
        # Load SCZ
        for i in range(f[scz_key].shape[0]):
            A = f[scz_key][i].copy()
            np.fill_diagonal(A, 0)
            
            # Apply threshold
            A[np.abs(A) < threshold] = 0.0
            
            scz_graphs.append(nx.from_numpy_array(A))
            
    return hc_graphs, scz_graphs


# In[ ]:


file_path = "./SMA_data_processing/cobre_combined_connectomes_database.h5"

print("BENCHMARK RESULTS")

for connectivity, hc_key, scz_key in [
    ("Pearson", "hc_pearson", "scz_pearson"),
    ("Glasso",  "hc_glasso", "scz_glasso"),
]:
    print(f"\nLoading {connectivity} Data")
    # graphs_hc, graphs_scz = load_graphs(file_path, hc_key, scz_key)
    graphs_hc, graphs_scz = load_graphs(file_path, hc_key, scz_key, threshold=0.3)
    print(f"HC {connectivity}: {len(graphs_hc)}  |  SCZ {connectivity}: {len(graphs_scz)}")

    print(f"\nBenchmarking {connectivity}")

    print(f"\n[LOUVAIN (NetworkX) — {connectivity}]")
    run_and_report(f"NX-{connectivity}", graphs_hc, graphs_scz, run_nx_louvain)

    print(f"\n[LOUVAIN (Custom)   — {connectivity}]")
    run_and_report(f"CL-{connectivity}", graphs_hc, graphs_scz, run_custom_louvain)

    print(f"\n[LEIDEN  (Custom)   — {connectivity}]")
    run_and_report(f"LD-{connectivity}", graphs_hc, graphs_scz, run_custom_leiden)

    print(f"\nClearing RAM for {connectivity}")
    del graphs_hc
    del graphs_scz
    gc.collect()



# In[ ]:




