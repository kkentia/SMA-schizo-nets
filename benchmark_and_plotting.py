import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import h5py
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from concurrent.futures import ProcessPoolExecutor

# Ensure the current script directory is on sys.path for worker subprocesses importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Leiden import leiden_algorithm, louvain_algorithm, modularity_vectorized

def evaluate_single_run(matrix, algo_name, max_iter, k_std = 1.0):
    """
    Takes a matrix as input, calculates the dynamic threshold, 
    generates the ground truth on the fly, and executes the tested algorithm.

    """
    A = np.abs(matrix.copy())
    np.fill_diagonal(A, 0)

    # Extracting the strict upper triangle to calculate the actual distribution (without duplicates)
    tri_upper = A[np.triu_indices_from(A, k=1)]
    mu = np.mean(tri_upper)
    sigma = np.std(tri_upper)

    # Adaptive dynamic thresholding formula based on the weight histogram
    threshold = max(0.0, mu - k_std * sigma)

    # Apply the statistic filtering
    A[A < threshold] = 0
    G = nx.from_numpy_array(A)

    if G.number_of_edges() == 0:
        return {"algo": algo_name, "iterations": max_iter, "runtime": 0.0, "modularity": 0.0, "nmi": 0.0}
    

    # DYNAMIC GROUND TRUTH FIX:
    # Official NetworkX Louvain reference is now computed on the fly 
    # for EACH subject. This eliminates the previous statistical bias 
    # (using Subject 1 as a global reference) and ensures a mathematically 
    # rigorous NMI score tailored to each individual connectome's topology.

    # Calculation of dynamic ground truth (specific to this particular patient)
    ref_comm_list = nx.algorithms.community.louvain_communities(G, seed=42)
    ref_labels = np.zeros(G.number_of_nodes())
    for comm_idx, comm in enumerate(ref_comm_list):
        for node in comm:
            ref_labels[node] = comm_idx
    
    # Execution and timing of the tested algorithm
    start_time = time.perf_counter()
    if algo_name.lower() == "louvain":
        partition = louvain_algorithm(G, max_levels=max_iter)
    elif algo_name.lower() == "leiden":
        partition = leiden_algorithm(G, max_iterations=max_iter)
    else: 
        raise ValueError(f"Unknown algorithm name: {algo_name}")
    runtime = time.perf_counter() - start_time

    # Calculation of metrics
    q_score = modularity_vectorized(G, partition)

    # Converting community dictionaries into a node-aligned vector for NMI
    nodes = sorted(list(G.nodes()))
    pred_labels = [partition[n] for n in nodes]
    nmi_score = normalized_mutual_info_score(ref_labels, pred_labels)

    return {"algo": algo_name, "iterations": max_iter, "runtime": runtime, "modularity": q_score, "nmi": nmi_score}

def run_benchmark_parallel(file_path, use_group_average=False, num_subjects=5, k_std=1.0, iterations_list=[1, 2, 5, 10, 20, 50]):
    """
    Loops through all groups (Sains vs SCZ), all connectivity options,
    handles individual samples or backup mode (average) and sends the raw matrices in parallel.

    """
    datasets= ["hc_pearson", "scz_pearson", "hc_glasso", "scz_glasso"]
    all_results = []

    with h5py.File(file_path, 'r') as f:
        for ds_name in datasets:
            print(f"Extraction and preparation of the set: {ds_name}...")
            data_cube = f[ds_name][:]

            # --- BACKUP FUNCTION (Average) vs INDIVIDUAL MODE ---
            if use_group_average:
                print(f" -> [BACKUP] Average connectome mode enabled for {ds_name}.")
                matrices_to_test = [np.mean(data_cube, axis=0)]
            else:
                print(f"-> [INDIVIDUAL] Analysis of the first {num_subjects} patients from {ds_name}.")
                matrices_to_test = data_cube[:num_subjects]

            # Preparing the data for transmission to the processors
            tasks = []
            for algo in ["louvain", "leiden"]:
                for iters in iterations_list:
                    for matrix in matrices_to_test:
                        tasks.append((matrix, algo, iters, k_std))

            """ for parralel processing, use that instead. 
            print(f" -> Parallel computing on the cluster for {ds_name}...")
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(evaluate_single_run, *task) for task in tasks]
                for fut in futures:
                    res = fut.result()
                    res["dataset"] = ds_name
                    all_results.append(res)
                    
            """

            print(f" -> Sequential computing (Safe Mode) for {ds_name}...")
            for task in tasks:
                # task contient : (matrix, algo, iters, k_std)
                res = evaluate_single_run(*task)
                res["dataset"] = ds_name
                all_results.append(res)

    df_res = pd.DataFrame(all_results)
    df_res.to_csv("benchmark_results_complete.csv", index=False)
    print("Benchmarking completed. Results saved to 'benchmark_results_complete.csv'.")

    # Averaging to smooth out inter-subject variability in the final graph
    df_summary = df_res.groupby(["dataset", "algo", "iterations"]).mean().reset_index()
    
    #generate_plots
    generate_plots(df_summary)

def generate_plots(df):
    """
    Generates two side-by-side figures (Leuven vs Leiden) per subgroup:
    - One figure for Quality (Q-Score) / Runtime / Iterations
    - One figure for Stability (NMI) / Runtime / Iterations
    """
    datasets = df["dataset"].unique()
    color_time = 'tab:red'

    for ds in datasets:
        df_ds = df[df["dataset"] == ds]
        df_louvain = df_ds[df_ds["algo"] == "louvain"].sort_values("iterations")
        df_leiden = df_ds[df_ds["algo"] == "leiden"].sort_values("iterations")

        # ─── FIGURE A: QUALITY (Q-SCORE) VS RUNTIME ─────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
        color_q = 'tab:blue'

        # Louvain (Left)
        axes[0].set_xlabel("Number of Iterations", fontweight='bold')
        axes[0].set_ylabel("Quality (Q-Score Modularity)", color=color_q, fontweight='bold')
        l1 = axes[0].plot(df_louvain["iterations"], df_louvain["modularity"], color=color_q, marker='o', linewidth=2.5 ,label='Quality(Q)')
        axes[0].tick_params(axis='y', labelcolor=color_q)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        ax0_twin = axes[0].twinx()
        ax0_twin.set_ylabel("Runtime (seconds)", color=color_time, fontweight='bold')
        l2 = ax0_twin.plot(df_louvain["iterations"], df_louvain["runtime"], color=color_time, marker='s', linewidth=2, label='Time (sec)')
        ax0_twin.tick_params(axis='y', labelcolor=color_time)
        axes[0].set_title("Louvain : Q-Score vs Runtime", fontweight='bold')
        axes[0].legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='lower right')

        # Leiden (Right)
        axes[1].set_xlabel("Number of Iterations", fontweight='bold')
        axes[1].set_ylabel("Quality (Q-Score Modularity)", color=color_q, fontweight='bold')
        l3 = axes[1].plot(df_leiden["iterations"], df_leiden["modularity"], color=color_q, marker='o', linewidth=2.5, label='Quality(Q)')
        axes[1].tick_params(axis='y', labelcolor=color_q)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        ax1_twin = axes[1].twinx()
        ax1_twin.set_ylabel("Runtime (seconds)", color=color_time, fontweight='bold')
        l4 = ax1_twin.plot(df_leiden["iterations"], df_leiden["runtime"], color=color_time, marker='s', linestyle = '--', linewidth=2, label='Time (sec)')
        ax1_twin.tick_params(axis='y', labelcolor=color_time)
        axes[1].set_title("Leiden : Q-Score vs Runtime", fontweight='bold')
        axes[1].legend(l3 + l4, [l.get_label() for l in l3 + l4], loc='lower right')

        plt.suptitle(f"Inflexion Quality Analysis: {ds.upper()}", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"benchmark_quality_{ds}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ─── FIGURE B: STABILITY (NMI) VS RUNTIME ───────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
        color_nmi = 'tab:green'

        # Louvain (Left)
        axes[0].set_xlabel("Number of Iterations", fontweight='bold')
        axes[0].set_ylabel("Stability (NMI)", color=color_nmi, fontweight='bold')
        l1 = axes[0].plot(df_louvain["iterations"], df_louvain["nmi"], color=color_nmi, marker='^', linewidth=2.5, label='Similarity (NMI)')
        axes[0].tick_params(axis='y', labelcolor=color_nmi)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        ax0_twin = axes[0].twinx()
        ax0_twin.set_ylabel("Runtime (seconds)", color=color_time, fontweight='bold')
        l2 = ax0_twin.plot(df_louvain["iterations"], df_louvain["runtime"], color=color_time, marker='s', linestyle='--', linewidth=2, label='Time (sec)')
        ax0_twin.tick_params(axis='y', labelcolor=color_time)
        axes[0].set_title("Louvain : NMI vs Runtime", fontweight='bold')
        axes[0].legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='lower right')

        # Leiden (Right)
        axes[1].set_xlabel("Number of Iterations", fontweight='bold')
        axes[1].set_ylabel("Stability (NMI)", color=color_nmi, fontweight='bold')
        l3 = axes[1].plot(df_leiden["iterations"], df_leiden["nmi"], color=color_nmi, marker='^', linewidth=2.5, label='Similarity (NMI)')
        axes[1].tick_params(axis='y', labelcolor=color_nmi)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        ax1_twin = axes[1].twinx()
        ax1_twin.set_ylabel("Runtime (seconds)", color=color_time, fontweight='bold')
        l4 = ax1_twin.plot(df_leiden["iterations"], df_leiden["runtime"], color=color_time, marker='s', linestyle='--', linewidth=2, label='Time (sec)')
        ax1_twin.tick_params(axis='y', labelcolor=color_time)
        axes[1].set_title("Leiden : NMI vs Runtime", fontweight='bold')
        axes[1].legend(l3 + l4, [l.get_label() for l in l3 + l4], loc='lower right') 

        plt.suptitle(f"Inflexion Stability Analysis: {ds.upper()}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"benchmark_stability_{ds}.png", dpi=300, bbox_inches='tight')
        plt.close()


    print("All plots generated and saved successfully.")

if __name__ == "__main__":
    file_path = "./SMA_data_processing/cobre_combined_connectomes_database.h5"

    datasets = ["hc_pearson", "scz_pearson", "hc_glasso", "scz_glasso"]

    # RUN CHECK:
    # If the server freezes or takes too long with the individual analysis, 
    # The user simply needs to set “use_group_average=True” to activate the backup immediately.

    run_benchmark_parallel(
    file_path=file_path,
    use_group_average=False,  # Set to True for backup mode (average connectome)
    num_subjects=5,          # Number of individual subjects to analyze (ignored if use
    k_std=1.0,              # Standard deviation multiplier for dynamic thresholding
    iterations_list=[1, 2, 5, 10, 20, 50]  # List of iteration counts to test
    )