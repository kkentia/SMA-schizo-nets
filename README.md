# SMA-schizo-nets

> **Network Analysis of Schizophrenic Brain Connectivity Using Community Detection**

| Name | Github Handle |
| --- | --- |
| Ana Bog | @kkentia |
| Yannick Künzli | @YannickKunz |
| Yann Gourraud | @Dace23 |
| Mathilde Voyame | @matvoyame |

---
See the [full project report](SMA_Project_Report.pdf) for details.
## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Outputs](#outputs)
- [Resources](#resources)
- [Task Distribution](#task-distribution)

---

## Project Description

We are using network graph theory to understand the human brain, which can be treated as a complex and modular IT system. Using preprocessed fMRI time series data from the COBRE dataset, we aim to build weighted network graphs where each node represents a brain region and edges represent functional connectivity between the nodes.

### Biological Problem

In the normal human brain, these networks function like well-encapsulated microservices. For example, the "Executive Network" (focusing) and the "Default Mode Network" (daydreaming) function independently. However, in Schizophrenia, this network encapsulation breaks down. The boundaries between these networks get blurred, and they start to "cross-wire," causing the brain to misinterpret internal thoughts as external hallucinations. 

### Computational & Analytical Approach

To quantitatively prove this structural breakdown of the Schizophrenia network, and to identify the specific topological causes, our approach will be to:

*   **Run Community Detection:** Use the State-of-the-Art Leiden algorithm to run community detection on our brain network graphs.
*   **Measure Modularity (Q Score):** Use this to calculate the Q Score of the network to understand how strictly separated these communities are. We expect a statistically significant lower Q Score in Schizophrenic networks compared to Healthy ones.
*   **Identify "Malicious Bridges" (Network Exploration):** Instead of just looking at the global breakdown, we will calculate node-level metrics, specifically the **Participation Coefficient**, to pinpoint the exact brain regions (nodes) that are inappropriately communicating outside of their designated communities. These are the "leaky valves" causing the cross-wiring.
*   **In Silico Network "Healing" (Targeted Simulation):** We will digitally delete these highly cross-wired "malicious bridge" nodes from the Schizophrenic networks and recalculate the Q-Score. Our aim is to test if the targeted removal of these faulty nodes mathematically restores the network's normal modular encapsulation.
*   **Visualization of the Breakdown and Simulation:** We will create colored and interactive 3D network graphs. The aim is to visually show how the nodes that normally form a neat community are scattered, explicitly highlight the "malicious bridges" in red, and chart the recovery trajectory of the Q-score after our simulation.

---

## Installation

### Prerequisites

- **Python 3.10+** (tested with 3.12 and 3.14)
- **Git**
- **pip** (comes with Python)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/kkentia/SMA-schizo-nets.git
cd SMA-schizo-nets
```

### Step 2 — Create a Virtual Environment

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

</details>

<details>
<summary><strong>Windows (Command Prompt)</strong></summary>

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Note:** If you get an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

</details>

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `networkx` | Graph data structures and library community detection |
| `numpy` | Numerical computation |
| `scipy` | Statistical tests (Welch's t-test) |
| `scikit-learn` | NMI scores, Graphical Lasso, StandardScaler |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical visualizations |
| `plotly` | Interactive 3D brain plots |
| `pandas` | DataFrames for results |
| `h5py` | HDF5 database I/O |
| `nibabel` | Neuroimaging file formats (.nii.gz) |
| `nilearn` | Brain atlas, masking, fMRI plotting |
| `tqdm` | Progress bars |
| `jupyter` | Notebook interface |
| `ipywidgets` | Interactive widgets in notebooks |

### Step 4 — Verify Installation

```bash
python -c "import networkx; import h5py; import nilearn; print('All dependencies OK')"
```

---

## Dataset Setup

This project uses the **COBRE (Center of Biomedical Research Excellence)** fMRI dataset.

### Option A — Use Pre-processed Connectomes (Recommended)

1. **Download** the pre-computed connectome database (~580 MB):

   📥 [**Download `cobre_combined_connectomes_database.h5` from Google Drive**](https://drive.google.com/file/d/1fq6UmiMmVRzqMvcFz9ZWia2TtlZuUAtm/view?usp=sharing)

2. Place it at:

```
SMA-schizo-nets/
└── SMA_data_processing/
    └── cobre_combined_connectomes_database.h5
```

> **Note:** This file is **not tracked by Git** due to its size. If the link above doesn't work, contact a team member or generate it from raw data (Very long runtime).

### Option B — Generate from Raw fMRI Data

1. Download the raw dataset from [Mendeley Data](https://data.mendeley.com/datasets/3h4mt7xryk/1)
2. Extract the `.nii.gz` files into `./data/1160600/`
3. Place the clinical CSV (`cobre_model_group.csv`) in the same directory
4. Run the preprocessing notebook:

```bash
jupyter notebook SMA_data_processing/01_exploration_masquage.ipynb
```

This will:
- Load raw fMRI volumes
- Apply the Schaefer 1000-parcel atlas (cortical) + Harvard-Oxford atlas (subcortical) → 1019 ROIs
- Compute Pearson correlation and Graphical Lasso connectivity matrices
- Save everything into `cobre_combined_connectomes_database.h5`

---

## Project Structure

```
SMA-schizo-nets/
│
├── src/
│   ├── Leiden.py                      # Custom Leiden & Louvain algorithm implementations
│   └── Louvain.py                     # Standalone Louvain implementation + initial analysis
│
├── benchmarks/
│   ├── benchmark.py                   # Benchmark: Custom Leiden vs Louvain vs NX Louvain (Q-score + p-values)
│   ├── benchmark_and_plotting.py      # Full parallel benchmark pipeline (Q, NMI, runtime) + plot generation
│   └── benchmark_enhanced.py          # Enhanced benchmark with extra metrics (Jaccard, etc.)
│
├── notebooks/
│   ├── leiden_visualizations.ipynb    # Main analysis notebook (community detection, bridge nodes, healing sim)
│   ├── louvain_visualizations.ipynb   # Louvain-specific analysis notebook
│   ├── Leiden.ipynb                   # Notebook version of Leiden.py
│   ├── Louvain.ipynb                  # Notebook version of Louvain.py
│   └── benchmark.ipynb                # Notebook version of benchmark.py
│
├── SMA_data_processing/
│   ├── 01_exploration_masquage.ipynb  # Data preprocessing pipeline
│   ├── 01_exploration_masquage.py     # Script version of preprocessing
│   └── cobre_combined_connectomes_database.h5  # Generated HDF5 database (not in Git)
│
├── results/
│   ├── benchmark_*.png                # Generated benchmark plots
│   ├── benchmark_results*.csv         # Benchmark CSV outputs
│   ├── fig*.png                       # Generated visualizations
│   ├── leiden_visualizations/         # More plots from Leiden analysis
│   └── louvain_visualizations/        # More plots from Louvain analysis
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

## Usage

> **Important:** Always activate your virtual environment first before running any command.

### 1. Community Detection & Full Analysis

Open the main analysis notebook:

```bash
jupyter notebook leiden_visualizations.ipynb
```

This runs the full pipeline:
- Loads the HDF5 connectome database
- Runs Leiden on all HC and SCZ subjects (Pearson & Glasso)
- Computes Q-scores, NMI, ARI with statistical tests
- Generates Q-score distribution plots, 3D brain maps, bridge node charts
- Runs the healing simulation

### 2. Algorithm Benchmarking

Run the benchmark comparing custom Leiden vs custom Louvain across iteration counts:

```bash
python benchmarks/benchmark_and_plotting.py
```

This will:
- Test both algorithms on 4 datasets × 5 subjects × 6 iteration counts
- Save results to `benchmark_results_complete.csv`
- Generate quality and stability plots for each dataset

For a quick comparison including NetworkX's library Louvain:

```bash
python benchmarks/benchmark.py
```

For the comprehensive validation benchmark (including Jaccard index, community overlap analysis, Pearson normalization, and the `leidenalg` C++ library):

```bash
python benchmarks/benchmark_enhanced.py
```

This generates `benchmark_enhanced_results.csv` and multiple comparison plots (`benchmark_overlay_*.png`, `benchmark_jaccard_*.png`, etc.).

### 3. Data Preprocessing (only needed if regenerating the database)

```bash
jupyter notebook SMA_data_processing/01_exploration_masquage.ipynb
```

---

## Outputs

### Benchmark Results (CSV)

| File | Description |
|---|---|
| `benchmark_enhanced_results.csv` | Enhanced benchmark comparing Custom vs Library algorithms (Q, NMI, Jaccard, within/between ratio) |
| `benchmark_results_complete.csv` | Full benchmark: 240 rows (algo × iterations × subjects × datasets) |
| `benchmark_results.csv` | Early single-subject prototype results |

### Generated Plots

| Plot | Description |
|---|---|
| `benchmark_quality_{dataset}.png` | Q-Score & Runtime vs Iterations (Louvain \| Leiden) |
| `benchmark_stability_{dataset}.png` | NMI & Runtime vs Iterations (Louvain \| Leiden) |
| `fig1_q_distribution_dual.png` | HC vs SCZ modularity violin/swarm plots |
| `fig3_bridge_nodes_dual.png` | Top-15 bridge ROIs bar chart (Glasso vs Pearson) |
| `fig_per_subject_q.png` | Per-subject Q-score comparison |
| `fig_healing_curves_dual.png` | Q-score recovery after progressive bridge removal |
| `fig2_brain_communities_*.png` | 3D brain community visualizations |

---

## Resources

*   **Relevant studies:**
    - [Classifying schizophrenic and controls from fMRI data using graph theoretic framework and community detection](https://link.springer.com/article/10.1007/s13721-023-00415-4) *(Note: our project differentiates itself by focusing on topological simulation and specific sub-network breakdowns rather than binary Machine Learning classification).*
    - [Nodal centrality of functional network in the differentiation of schizophrenia](https://pubmed.ncbi.nlm.nih.gov/26299706/)
    - [Detecting schizophrenia at the level of the individual: relative diagnostic value of whole brain images, connectome-wide functional connectivity and graph-based metrics](https://pubmed.ncbi.nlm.nih.gov/31391132/)
    - [Decreased small-world functional network connectivity and clustering across resting state networks in schizophrenia: an fMRI classification tutorial](https://pubmed.ncbi.nlm.nih.gov/24032010/)
    - [Hierarchical network disruptions in Schizophrenia: A multi-level fMRI study of functional connectivity](https://pubmed.ncbi.nlm.nih.gov/41110182/)
    - [Restoring Synaptic Balance in Schizophrenia: Insights From a Thalamo-Cortical Conductance-Based Model](https://academic.oup.com/schizophreniabulletin/advance-article/doi/10.1093/schbul/sbaf149/8250824)
    - [Revealing multiple biological subtypes of schizophrenia through a data-driven approach](https://link.springer.com/article/10.1186/s12967-025-06503-5) *(Note: relevant from page 11 on DMN and SMN)*
      
* [Preprocessed Dataset](https://data.mendeley.com/datasets/3h4mt7xryk/1)  
* [Original Dataset](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html) 

---

## Task Distribution

*   **Ana Bog:** Analysis of the results & visualizations (generating the 3D brain maps via `nilearn`, highlighting the bridge nodes, and creating the "healing" Q-score comparison charts).
*   **Yannick Künzli:** Core implementations (calculating the Participation Coefficient, identifying the malicious bridges, and building the node deletion simulation loop).
*   **Yann Gourraud:** Core implementations (running the baseline Leiden algorithm, calculating the initial Modularity Q-Scores, and performing statistical validations between groups).
*   **Mathilde Voyame:** Data processing & result analysis (ingesting the `.nii.gz` files, applying the brain atlas to extract the network matrices, and researching the biological function of our identified nodes). 

*Additionally, everyone will do their own literature research to support their specific tasks.*

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
