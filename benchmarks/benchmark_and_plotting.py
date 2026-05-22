import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import h5py
from nilearn import datasets
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from concurrent.futures import ProcessPoolExecutor

# Ensure the parent directory is on sys.path for worker subprocesses importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Leiden import leiden_algorithm, louvain_algorithm, modularity_vectorized


def compute_threshold_and_filter(matrix, k_std=1.0):
    """
    Calculates the adaptive statistical threshold based on the distribution
    of upper-triangle weights and eliminates weak connections.
    """
    A = np.abs(matrix.copy())
    np.fill_diagonal(A, 0)

    # Extract strict upper triangle to compute accurate distribution metrics
    tri_upper = A[np.triu_indices_from(A, k=1)]
    mu = np.mean(tri_upper)
    sigma = np.std(tri_upper)

    # Apply adaptive thresholding formula
    threshold = max(0.0, mu - k_std * sigma)
    A[A < threshold] = 0
    return A, threshold


def inspect_and_save_matrix(matrix, threshold_val, title="Matrix"):
    """
    Calculates network density, mean node degree (excluding diagonal),
    and saves a structural visualization plot (Sparsity pattern).
    """
    mat_no_diag = np.abs(matrix.copy())
    np.fill_diagonal(mat_no_diag, 0)

    # 1. Edge Count and Density Calculations
    n_edges = np.count_nonzero(mat_no_diag) / 2
    n_nodes = matrix.shape[0]
    max_edges = (n_nodes * (n_nodes - 1)) / 2
    density = (n_edges / max_edges) * 100

    # 2. Mean Node Degree Calculation
    adj_matrix = (mat_no_diag > 0).astype(int)
    node_degrees = np.sum(adj_matrix, axis=1)
    mean_degree = np.mean(node_degrees)

    print(f"--- AUTOMATIC DIAGNOSTIC: {title} ---")
    print(f"Node Count          : {n_nodes}")
    print(f"Applied Threshold   : {threshold_val:.4f}")
    print(f"Active Edges        : {int(n_edges)} / {int(max_edges)}")
    print(f"Network Density     : {density:.4f}%")
    print(f"Mean Degree per Node: {mean_degree:.2f} neighbors / node\n")

    # 3. Sparsity Plot Generation
    plt.figure(figsize=(6, 6))
    plt.spy(mat_no_diag, markersize=0.5, color='black') 
    plt.title(f"{title}\nNodes: {n_nodes} | K-Mean: {mean_degree:.1f} | Density: {density:.2f}%", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"diagnostic_sparsity_{title.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()


def evaluate_single_run(thresholded_matrix, algo_name, max_iter):
    """
    Executes the targeted community detection algorithm on a pre-filtered matrix
    and generates the dynamic baseline ground truth on the fly.
    """
    G = nx.from_numpy_array(thresholded_matrix)

    if G.number_of_edges() == 0:
        return {"algo": algo_name, "iterations": max_iter, "runtime": 0.0, "modularity": 0.0, "nmi": 0.0}

    # Generate localized individual ground truth reference
    ref_comm_list = nx.algorithms.community.louvain_communities(G, seed=42)
    ref_labels = np.zeros(G.number_of_nodes())
    for comm_idx, comm in enumerate(ref_comm_list):
        for node in comm:
            ref_labels[node] = comm_idx
    
    # Execution and timing metrics
    start_time = time.perf_counter()
    if algo_name.lower() == "louvain":
        partition = louvain_algorithm(G, max_levels=max_iter)
    elif algo_name.lower() == "leiden":
        partition = leiden_algorithm(G, max_iterations=max_iter)
    else: 
        raise ValueError(f"Unknown algorithm name: {algo_name}")
    runtime = time.perf_counter() - start_time

    # Calculate Modularity and Normalised Mutual Information
    q_score = modularity_vectorized(G, partition)
    nodes = sorted(list(G.nodes()))
    pred_labels = [partition[n] for n in nodes]
    nmi_score = normalized_mutual_info_score(ref_labels, pred_labels)

    return {"algo": algo_name, "iterations": max_iter, "runtime": runtime, "modularity": q_score, "nmi": nmi_score}


def run_benchmark_parallel(file_path, valid_roi_mask, use_group_average=False, num_subjects=5, k_std=1.0, iterations_list=[1, 2, 5, 10, 20, 50]):
    """
    Benchmarks community structure tracking across connectivity paradigms
    while safely maintaining isolation of clinical subgroups.
    """
    datasets_list = ["hc_pearson", "scz_pearson", "hc_glasso", "scz_glasso"]
    all_results = []

    with h5py.File(file_path, 'r') as f:

        print("="*60)
        print("STARTING NETWORK MATRIX ANATOMICAL INSPECTION (THRESHOLDED 1017 NODES)")
        print("="*60)

        for ds_name in datasets_list:
            raw_cube = f[ds_name][:]
            
            # 1. Apply Live anatomical Masking (1019 -> 1017)
            clean_cube = raw_cube[:, valid_roi_mask][:, :, valid_roi_mask]
            
            # 2. Isolate target structural sample
            sample_matrix = clean_cube[0] if not use_group_average else np.mean(clean_cube, axis=0)
            prefix = "Average" if use_group_average else "Subject_1"
            
            # 3. Apply Thresholding BEFORE diagnostics to accurately observe non-100% density
            thresholdED_sample, target_thresh = compute_threshold_and_filter(sample_matrix, k_std=k_std)
            
            # Run diagnostic evaluation and overwrite old figures
            inspect_and_save_matrix(thresholdED_sample, threshold_val=target_thresh, title=f"{ds_name.upper()}_{prefix}")
            
        print("="*60)
        print(" INSPECTION COMPLETE. All thresholded diagnostic plots have been overwritten.")
        print(" [STOP EXPRESS MODE] Intentional script termination before parallel compute.")
        print("="*60)
        return  # Comment out this line when ready to launch the complete loop calculation pipeline!

        # ─── PRODUCTION EXECUTION LOOP ───────────────────────────────────────
        for ds_name in datasets_list:
            print(f"Extraction and preparation of the set: {ds_name}...")
            raw_cube = f[ds_name][:]
            clean_cube = raw_cube[:, valid_roi_mask][:, :, valid_roi_mask]

            if use_group_average:
                print(f" -> [BACKUP] Average connectome mode enabled for {ds_name}.")
                matrices_to_test = [np.mean(clean_cube, axis=0)]
            else:
                print(f"-> [INDIVIDUAL] Analysis of the first {num_subjects} patients from {ds_name}.")
                matrices_to_test = clean_cube[:num_subjects]

            # Pre-filter all matrices across the threshold criteria
            thresholded_matrices = []
            for mat in matrices_to_test:
                thresh_mat, _ = compute_threshold_and_filter(mat, k_std=k_std)
                thresholded_matrices.append(thresh_mat)

            # Package computational tasks
            tasks = []
            for algo in ["louvain", "leiden"]:
                for iters in iterations_list:
                    for matrix in matrices_to_test:
                        tasks.append((matrix, algo, iters, k_std))

            num_workers = min(os.cpu_count() or 4, 6)  # use up to 6 cores, leave headroom
            print(f" -> Parallel computing ({num_workers} workers) for {ds_name}...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(evaluate_single_run, *task) for task in tasks]
                for fut in futures:
                    res = fut.result()
                    res["dataset"] = ds_name
                    all_results.append(res)

    df_res = pd.DataFrame(all_results)
    df_res.to_csv("benchmark_results_complete.csv", index=False)
    print("Benchmarking completed. Results saved to 'benchmark_results_complete.csv'.")

    df_summary = df_res.groupby(["dataset", "algo", "iterations"]).mean(numeric_only=True).reset_index()
    generate_plots(df_summary)


def generate_plots(df):
    """
    Generates side-by-side diagnostic figures for Quality and Stability benchmarks.
    """
    if df.empty:
        return
    datasets_avail = df["dataset"].unique()
    color_time = 'tab:red'

    for ds in datasets_avail:
        df_ds = df[df["dataset"] == ds]
        df_louvain = df_ds[df_ds["algo"] == "louvain"].sort_values("iterations")
        df_leiden = df_ds[df_ds["algo"] == "leiden"].sort_values("iterations")

        # FIGURE A: QUALITY (Q-SCORE) VS RUNTIME
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
        color_q = 'tab:blue'

        # Louvain (Left)
        axes[0].set_xlabel("Number of Iterations", fontweight='bold')
        axes[0].set_ylabel("Quality (Q-Score Modularity)", color=color_q, fontweight='bold')
        l1 = axes[0].plot(df_louvain["iterations"], df_louvain["modularity"], color=color_q, marker='o', linewidth=2.5, label='Quality(Q)')
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
        l4 = ax1_twin.plot(df_leiden["iterations"], df_leiden["runtime"], color=color_time, marker='s', linestyle='--', linewidth=2, label='Time (sec)')
        ax1_twin.tick_params(axis='y', labelcolor=color_time)
        axes[1].set_title("Leiden : Q-Score vs Runtime", fontweight='bold')
        axes[1].legend(l3 + l4, [l.get_label() for l in l3 + l4], loc='lower right')

        plt.suptitle(f"Inflexion Quality Analysis: {ds.upper()}", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"benchmark_quality_{ds}.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SMA_data_processing", "cobre_combined_connectomes_database.h5"))

    # 1. Fetch Atlas structures exactly like your friend
    atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2)
    atlas_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

    schaefer_labels = [l.decode() if isinstance(l, bytes) else l for l in atlas_schaefer.labels][1:]
    ho_labels_raw = [l.decode() if isinstance(l, bytes) else l for l in atlas_ho.labels]

    # Load HO data and apply your friend's exact zeroing method
    # This is what drops Left White Matter (1) and Left Cortex (2)
    import nibabel as nib
    ho_img = nib.load(atlas_ho.maps) if isinstance(atlas_ho.maps, str) else atlas_ho.maps
    ho_data = ho_img.get_fdata().copy()
    
    ho_data[ho_data == 1] = 0
    ho_data[ho_data == 2] = 0

    # Dynamically extract the remaining 19 unique subcortical label IDs (typically 3 through 21)
    ho_unique_labels = sorted([int(v) for v in np.unique(ho_data) if v > 0])

    # 2. Flag the remaining Right-Side Artifacts (Labels 3 and 4)
    # Note the spelling 'ARTEFACT' to match her exact text format
    ARTEFACT_NAMES = {'Right Cerebral White Matter', 'Right Cerebral Cortex'}

    ho_labels = []
    for lid in ho_unique_labels:
        name = ho_labels_raw[lid]
        if name in ARTEFACT_NAMES:
            ho_labels.append(f'[ARTEFACT] {name}')
        else:
            ho_labels.append(name)

    # 3. Concatenate and Build the 1019 Boolean Mask
    all_labels = schaefer_labels + ho_labels
    VALID_ROI_MASK = np.array(['[ARTEFACT]' not in l for l in all_labels])

    # Print structural tracking validation to console
    print("="*60)
    print("ATLAS LABELS PARSING DIAGNOSTIC (FRIEND'S ALIGNMENT METHOD)")
    print("="*60)
    print(f"Schaefer Cortical Count   : {len(schaefer_labels)}")
    print(f"HO Subcortical Count      : {len(ho_labels)} (Successfully dropped left side artifacts)")
    print(f"Total Mask Array Length   : {len(VALID_ROI_MASK)} nodes")
    print(f"Valid Remaining Brain ROIs: {VALID_ROI_MASK.sum()} / 1019")
    print(f"Identified Artifacts to Cut: {(~VALID_ROI_MASK).sum()} / 1019")
    
    # Show exactly where the right-side white matter and cortex artifacts live in the matrix columns
    artifact_indices = np.where(~VALID_ROI_MASK)[0]
    for idx in artifact_indices:
        print(f"  -> Found target artifact at index {idx}: '{all_labels[idx]}'")
    print("="*60 + "\n")

    # 4. Fire execution loop safely
    run_benchmark_parallel(
        file_path=file_path,
        valid_roi_mask=VALID_ROI_MASK,
        use_group_average=False,
        num_subjects=5,
        k_std=1.0,
        iterations_list=[1, 2, 5, 10, 20, 50]
    )