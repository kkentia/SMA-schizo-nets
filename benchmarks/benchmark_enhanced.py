"""
Enhanced Benchmark: Leiden vs Louvain vs Library Algorithms
==========================================================

Compares:
  - Custom Leiden (Leiden.py)
  - Custom Louvain (Leiden.py)
  - NetworkX Louvain (library)
  - leidenalg/igraph Leiden (library, if installed)

Metrics:
  - Q-Score (Modularity)
  - NMI (Normalized Mutual Information)
  - Jaccard Index (community partition similarity)
  - Runtime (seconds)
  - Community Overlap analysis
  - Pearson normalization by community/brain region

Outputs:
  - benchmark_enhanced_results.csv  (raw results)
  - benchmark_overlay_*.png          (overlay comparison plots)
  - benchmark_jaccard_*.png          (Jaccard heatmaps)
  - benchmark_community_overlap_*.png (community overlap analysis)
  - benchmark_pearson_norm_*.png     (Pearson normalization by community)
"""

import os
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import h5py
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")

# Ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Leiden import leiden_algorithm, louvain_algorithm, modularity_vectorized

# ──────────────────────────────────────────────────────────────────────
# Try importing leidenalg / igraph for library Leiden comparison
# ──────────────────────────────────────────────────────────────────────
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDENALG = True
    print("[INFO] leidenalg + igraph found. Library Leiden will be included.")
except ImportError:
    HAS_LEIDENALG = False
    print("[INFO] leidenalg/igraph not installed. Skipping library Leiden.")
    print("       Install with: pip install leidenalg python-igraph")


# ═══════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def jaccard_index(partition_a, partition_b):
    """
    Compute the Jaccard Index between two partitions.

    For every pair of nodes (i, j), check if they are co-classified
    (in the same community) in both partitions. The Jaccard index is:

        J = |S_a ∩ S_b| / |S_a ∪ S_b|

    where S_a is the set of node pairs co-classified in partition A,
    and S_b is the set of node pairs co-classified in partition B.
    """
    nodes = sorted(set(partition_a.keys()) & set(partition_b.keys()))
    if len(nodes) < 2:
        return 1.0

    agree = 0     # pairs co-classified in BOTH
    either = 0    # pairs co-classified in AT LEAST ONE

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            same_a = partition_a[ni] == partition_a[nj]
            same_b = partition_b[ni] == partition_b[nj]

            if same_a or same_b:
                either += 1
                if same_a and same_b:
                    agree += 1

    return agree / either if either > 0 else 1.0


def community_overlap_matrix(partition_a, partition_b):
    """
    Compute a confusion/overlap matrix between two partitions.

    Returns:
      overlap_df: DataFrame where rows are communities in partition_a,
                  columns are communities in partition_b, and values
                  are the number of shared nodes.
    """
    comms_a = sorted(set(partition_a.values()))
    comms_b = sorted(set(partition_b.values()))

    # Build sets
    sets_a = {c: set() for c in comms_a}
    sets_b = {c: set() for c in comms_b}
    for node, comm in partition_a.items():
        sets_a[comm].add(node)
    for node, comm in partition_b.items():
        sets_b[comm].add(node)

    matrix = np.zeros((len(comms_a), len(comms_b)), dtype=int)
    for i, ca in enumerate(comms_a):
        for j, cb in enumerate(comms_b):
            matrix[i, j] = len(sets_a[ca] & sets_b[cb])

    return pd.DataFrame(matrix,
                        index=[f"A_{c}" for c in comms_a],
                        columns=[f"B_{c}" for c in comms_b])


def pearson_by_community(matrix, partition, nodes):
    """
    Compute mean within-community and between-community correlations.

    Args:
        matrix:    NxN connectivity matrix (Pearson)
        partition: dict {node: community_id}
        nodes:     list of node indices

    Returns:
        within_mean:  mean correlation within communities
        between_mean: mean correlation between communities
        ratio:        within/between ratio (higher = more modular)
    """
    comms = defaultdict(list)
    for n in nodes:
        comms[partition[n]].append(n)

    within_vals = []
    between_vals = []

    for comm_nodes in comms.values():
        # Within-community pairs
        for i in range(len(comm_nodes)):
            for j in range(i + 1, len(comm_nodes)):
                within_vals.append(abs(matrix[comm_nodes[i], comm_nodes[j]]))

    comm_list = list(comms.values())
    for ci in range(len(comm_list)):
        for cj in range(ci + 1, len(comm_list)):
            for ni in comm_list[ci]:
                for nj in comm_list[cj]:
                    between_vals.append(abs(matrix[ni, nj]))

    within_mean = np.mean(within_vals) if within_vals else 0.0
    between_mean = np.mean(between_vals) if between_vals else 0.0
    ratio = within_mean / between_mean if between_mean > 0 else float('inf')

    return within_mean, between_mean, ratio


# ═══════════════════════════════════════════════════════════════════════
# ALGORITHM RUNNERS
# ═══════════════════════════════════════════════════════════════════════

def run_custom_leiden(G, max_iter=50):
    start = time.perf_counter()
    partition = leiden_algorithm(G, max_iterations=max_iter)
    runtime = time.perf_counter() - start
    return partition, runtime


def run_custom_louvain(G, max_levels=50):
    start = time.perf_counter()
    partition = louvain_algorithm(G, max_levels=max_levels)
    runtime = time.perf_counter() - start
    return partition, runtime


def run_nx_louvain(G):
    start = time.perf_counter()
    comm_list = nx.algorithms.community.louvain_communities(G, weight='weight', seed=42)
    runtime = time.perf_counter() - start
    # Convert to dict
    partition = {}
    for comm_idx, comm in enumerate(comm_list):
        for node in comm:
            partition[node] = comm_idx
    return partition, runtime


def run_library_leiden(G):
    """Run leidenalg library Leiden via igraph."""
    if not HAS_LEIDENALG:
        return None, 0.0

    # Convert NetworkX → igraph
    edges = list(G.edges())
    weights = [G[u][v].get('weight', 1.0) for u, v in edges]
    ig_G = ig.Graph(n=G.number_of_nodes(), edges=edges)
    ig_G.es['weight'] = weights

    start = time.perf_counter()
    result = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition,
                                      weights='weight', seed=42)
    runtime = time.perf_counter() - start

    # Convert to dict
    partition = {}
    for comm_idx, comm in enumerate(result):
        for node in comm:
            partition[node] = comm_idx
    return partition, runtime


# ═══════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

def build_graph(matrix, k_std=1.0, apply_threshold=True):
    """Threshold and build graph from a connectivity matrix."""
    A = np.abs(matrix.copy())
    np.fill_diagonal(A, 0)
    if apply_threshold:
        tri_upper = A[np.triu_indices_from(A, k=1)]
        mu = np.mean(tri_upper)
        sigma = np.std(tri_upper)
        threshold = max(0.0, mu - k_std * sigma)
        A[A < threshold] = 0
    return nx.from_numpy_array(A)


def run_enhanced_benchmark(file_path, num_subjects=5, k_std=1.0):
    """
    Run the full enhanced benchmark on all datasets.
    """
    datasets = ["hc_pearson", "scz_pearson", "hc_glasso", "scz_glasso"]
    all_results = []

    with h5py.File(file_path, 'r') as f:
        for ds_name in datasets:
            print(f"\n{'='*60}")
            print(f"  Dataset: {ds_name}")
            print(f"{'='*60}")

            data_cube = f[ds_name][:]
            matrices = data_cube[:num_subjects]
            apply_thresh = "glasso" not in ds_name.lower()

            for subj_idx, matrix in enumerate(matrices):
                print(f"  Subject {subj_idx + 1}/{num_subjects}...")
                G = build_graph(matrix, k_std, apply_threshold=apply_thresh)

                if G.number_of_edges() == 0:
                    print(f"    [SKIP] No edges after thresholding.")
                    continue

                nodes = sorted(list(G.nodes()))

                # ── Run all algorithms ──
                algorithms = {}
                algorithms["Custom Leiden"] = run_custom_leiden(G)
                algorithms["Custom Louvain"] = run_custom_louvain(G)
                algorithms["NX Louvain"] = run_nx_louvain(G)
                if HAS_LEIDENALG:
                    algorithms["Library Leiden"] = run_library_leiden(G)

                # ── Compute reference (NX Louvain) for NMI/Jaccard ──
                ref_partition = algorithms["NX Louvain"][0]
                ref_labels = [ref_partition[n] for n in nodes]

                for algo_name, (partition, runtime) in algorithms.items():
                    if partition is None:
                        continue

                    # Q-Score
                    q_score = modularity_vectorized(G, partition)

                    # NMI
                    pred_labels = [partition[n] for n in nodes]
                    nmi = normalized_mutual_info_score(ref_labels, pred_labels)

                    # Jaccard
                    jacc = jaccard_index(ref_partition, partition)

                    # Pearson normalization by community
                    within, between, ratio = pearson_by_community(
                        np.abs(matrix), partition, nodes
                    )

                    # Number of communities
                    n_communities = len(set(partition.values()))

                    all_results.append({
                        "dataset": ds_name,
                        "subject": subj_idx,
                        "algo": algo_name,
                        "modularity": q_score,
                        "nmi": nmi,
                        "jaccard": jacc,
                        "runtime": runtime,
                        "n_communities": n_communities,
                        "within_community_corr": within,
                        "between_community_corr": between,
                        "within_between_ratio": ratio,
                    })

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv("benchmark_enhanced_results.csv", index=False)
    print(f"\n[SAVED] benchmark_enhanced_results.csv ({len(df)} rows)")

    # Generate all plots
    generate_overlay_plots(df)
    generate_jaccard_heatmaps(df)
    generate_pearson_normalization_plots(df)
    generate_community_overlap_plots(file_path, num_subjects, k_std)

    return df


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

ALGO_COLORS = {
    "Custom Leiden":  "#2196F3",   # blue
    "Custom Louvain": "#FF9800",   # orange
    "NX Louvain":     "#4CAF50",   # green
    "Library Leiden":  "#9C27B0",  # purple
}

ALGO_MARKERS = {
    "Custom Leiden":  "o",
    "Custom Louvain": "s",
    "NX Louvain":     "^",
    "Library Leiden":  "D",
}


def generate_overlay_plots(df):
    """
    Generate overlay comparison plots: all algorithms on the same axes.
    One figure per dataset with 4 subplots: Q-Score, NMI, Jaccard, Runtime.
    """
    datasets = df["dataset"].unique()

    for ds in datasets:
        df_ds = df[df["dataset"] == ds]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Algorithm Comparison — {ds.upper()}", fontsize=16, fontweight='bold')

        metrics = [
            ("modularity", "Q-Score (Modularity)", axes[0, 0]),
            ("nmi", "NMI vs NX Louvain", axes[0, 1]),
            ("jaccard", "Jaccard Index vs NX Louvain", axes[1, 0]),
            ("runtime", "Runtime (seconds)", axes[1, 1]),
        ]

        for metric_col, metric_label, ax in metrics:
            for algo in df_ds["algo"].unique():
                df_algo = df_ds[df_ds["algo"] == algo]
                vals = df_algo[metric_col].values
                color = ALGO_COLORS.get(algo, "gray")

                # Bar-style grouped comparison
                positions = range(len(vals))
                ax.bar(
                    [p + list(df_ds["algo"].unique()).index(algo) * 0.2 for p in positions],
                    vals,
                    width=0.18,
                    color=color,
                    alpha=0.85,
                    label=algo
                )

            ax.set_ylabel(metric_label, fontweight='bold')
            ax.set_xlabel("Subject Index", fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"benchmark_overlay_{ds}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] benchmark_overlay_{ds}.png")


def generate_jaccard_heatmaps(df):
    """
    Generate Jaccard similarity heatmaps between all algorithm pairs.
    """
    datasets = df["dataset"].unique()

    for ds in datasets:
        df_ds = df[df["dataset"] == ds]
        algos = sorted(df_ds["algo"].unique())
        n_algos = len(algos)

        # Average Jaccard across subjects for each algo pair
        # We'll compute pairwise Jaccard from the existing data
        # Since Jaccard was computed vs NX Louvain reference,
        # we present the summary as a table

        fig, ax = plt.subplots(figsize=(8, 5))

        # Summary: per-algo average metrics
        summary_data = []
        for algo in algos:
            df_algo = df_ds[df_ds["algo"] == algo]
            summary_data.append({
                "Algorithm": algo,
                "Avg Q": f"{df_algo['modularity'].mean():.4f}",
                "Avg NMI": f"{df_algo['nmi'].mean():.4f}",
                "Avg Jaccard": f"{df_algo['jaccard'].mean():.4f}",
                "Avg Runtime": f"{df_algo['runtime'].mean():.4f}s",
                "Avg #Comms": f"{df_algo['n_communities'].mean():.1f}",
            })

        summary_df = pd.DataFrame(summary_data)

        ax.axis('off')
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for j in range(len(summary_df.columns)):
            table[0, j].set_facecolor('#2196F3')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Color algo column
        for i in range(len(algos)):
            algo = summary_data[i]["Algorithm"]
            color = ALGO_COLORS.get(algo, "#EEEEEE")
            table[i + 1, 0].set_facecolor(color)
            table[i + 1, 0].set_text_props(color='white', fontweight='bold')

        ax.set_title(f"Algorithm Summary — {ds.upper()}", fontsize=14,
                     fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f"benchmark_jaccard_{ds}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] benchmark_jaccard_{ds}.png")


def generate_pearson_normalization_plots(df):
    """
    Plot within-community vs between-community correlation strength.
    This shows how well each algorithm separates communities in terms
    of actual Pearson correlation strength.
    """
    datasets = df["dataset"].unique()

    for ds in datasets:
        df_ds = df[df["dataset"] == ds]
        algos = sorted(df_ds["algo"].unique())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f"Pearson Normalization by Community — {ds.upper()}",
                     fontsize=14, fontweight='bold')

        # Plot 1: Within vs Between correlation
        ax = axes[0]
        x = np.arange(len(algos))
        width = 0.35
        within_means = [df_ds[df_ds["algo"] == a]["within_community_corr"].mean() for a in algos]
        between_means = [df_ds[df_ds["algo"] == a]["between_community_corr"].mean() for a in algos]

        bars1 = ax.bar(x - width/2, within_means, width, label='Within-community',
                       color='#2196F3', alpha=0.85)
        bars2 = ax.bar(x + width/2, between_means, width, label='Between-community',
                       color='#FF5722', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(" ", "\n") for a in algos], fontsize=8)
        ax.set_ylabel("Mean |Correlation|", fontweight='bold')
        ax.set_title("Within vs Between Community", fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        # Plot 2: Within/Between ratio
        ax = axes[1]
        ratios = [df_ds[df_ds["algo"] == a]["within_between_ratio"].mean() for a in algos]
        colors = [ALGO_COLORS.get(a, "gray") for a in algos]
        ax.bar(x, ratios, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(" ", "\n") for a in algos], fontsize=8)
        ax.set_ylabel("Within / Between Ratio", fontweight='bold')
        ax.set_title("Community Separation Quality", fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ratio = 1 (no separation)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        # Plot 3: Number of communities detected
        ax = axes[2]
        n_comms = [df_ds[df_ds["algo"] == a]["n_communities"].mean() for a in algos]
        ax.bar(x, n_comms, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(" ", "\n") for a in algos], fontsize=8)
        ax.set_ylabel("# Communities (avg)", fontweight='bold')
        ax.set_title("Number of Communities Detected", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"benchmark_pearson_norm_{ds}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] benchmark_pearson_norm_{ds}.png")


def generate_community_overlap_plots(file_path, num_subjects=5, k_std=1.0):
    """
    For one representative subject per dataset, generate a community
    overlap confusion matrix between Leiden and Louvain partitions.
    """
    datasets_meta = [
        ("hc_glasso", "HC Glasso"),
        ("scz_glasso", "SCZ Glasso"),
        ("hc_pearson", "HC Pearson"),
        ("scz_pearson", "SCZ Pearson"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Community Overlap: Custom Leiden vs Custom Louvain (Subject 0)",
                 fontsize=14, fontweight='bold')

    with h5py.File(file_path, 'r') as f:
        for idx, (ds_name, ds_label) in enumerate(datasets_meta):
            ax = axes[idx // 2][idx % 2]
            matrix = f[ds_name][0]
            apply_thresh = "glasso" not in ds_name.lower()
            G = build_graph(matrix, k_std, apply_threshold=apply_thresh)

            if G.number_of_edges() == 0:
                ax.set_title(f"{ds_label} — No edges", fontweight='bold')
                continue

            part_leiden, _ = run_custom_leiden(G)
            part_louvain, _ = run_custom_louvain(G)

            overlap = community_overlap_matrix(part_leiden, part_louvain)

            # Only show top communities (max 15 each for readability)
            top_rows = overlap.sum(axis=1).nlargest(min(15, len(overlap))).index
            top_cols = overlap.sum(axis=0).nlargest(min(15, len(overlap.columns))).index
            overlap_sub = overlap.loc[top_rows, top_cols]

            im = ax.imshow(overlap_sub.values, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(overlap_sub.columns)))
            ax.set_xticklabels([c.replace("B_", "Louv ") for c in overlap_sub.columns],
                               rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(len(overlap_sub.index)))
            ax.set_yticklabels([r.replace("A_", "Leid ") for r in overlap_sub.index],
                               fontsize=7)
            ax.set_title(f"{ds_label}", fontweight='bold')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("benchmark_community_overlap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] benchmark_community_overlap.png")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SMA_data_processing", "cobre_combined_connectomes_database.h5"))

    if not os.path.exists(file_path):
        print(f"[ERROR] Database not found: {file_path}")
        print("        Run the data preprocessing notebook first.")
        sys.exit(1)

    print("=" * 60)
    print("  ENHANCED BENCHMARK: Leiden vs Louvain vs Libraries")
    print("=" * 60)

    df = run_enhanced_benchmark(
        file_path=file_path,
        num_subjects=5,
        k_std=1.0,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    summary = df.groupby(["dataset", "algo"]).agg({
        "modularity": "mean",
        "nmi": "mean",
        "jaccard": "mean",
        "runtime": "mean",
        "n_communities": "mean",
        "within_between_ratio": "mean",
    }).round(4)
    print(summary.to_string())
    print("\n[DONE] All plots and results saved.")
