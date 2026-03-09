# SMA-schizo-nets

| Name | Github Handle |
| --- | --- |
| Ana Bog | @kkentia |
| Yannick Künzli | @YannickKunz |
| Yann Gourraud | @Dace23 |
| Mathilde Voyame | @matvoyame |

# Project Description

We are using network graph theory to understand the human brain, which can be treated as a complex and modular IT system. Using preprocessed fMRI time series data from the COBRE dataset, we aim to build weighted network graphs where each node represents a brain region and edges represent functional connectivity between the nodes.

## Biological Problem

In the normal human brain, these networks function like well-encapsulated microservices. For example, the "Executive Network" (focusing) and the "Default Mode Network" (daydreaming) function independently. However, in Schizophrenia, this network encapsulation breaks down. The boundaries between these networks get blurred, and they start to "cross-wire," causing the brain to misinterpret internal thoughts as external hallucinations. 

## Computational & Analytical Approach

To quantitatively prove this structural breakdown of the Schizophrenia network, and to identify the specific topological causes, our approach will be to:

*   **Run Community Detection:** Use the State-of-the-Art Leiden algorithm to run community detection on our brain network graphs.
*   **Measure Modularity (Q Score):** Use this to calculate the Q Score of the network to understand how strictly separated these communities are. We expect a statistically significant lower Q Score in Schizophrenic networks compared to Healthy ones.
*   **Identify "Malicious Bridges" (Network Exploration):** Instead of just looking at the global breakdown, we will calculate node-level metrics, specifically the **Participation Coefficient**, to pinpoint the exact brain regions (nodes) that are inappropriately communicating outside of their designated communities. These are the "leaky valves" causing the cross-wiring.
*   **In Silico Network "Healing" (Targeted Simulation):** We will digitally delete these highly cross-wired "malicious bridge" nodes from the Schizophrenic networks and recalculate the Q-Score. Our aim is to test if the targeted removal of these faulty nodes mathematically restores the network's normal modular encapsulation.
*   **Visualization of the Breakdown and Simulation:** We will create colored and interactive 3D network graphs. The aim is to visually show how the nodes that normally form a neat community are scattered, explicitly highlight the "malicious bridges" in red, and chart the recovery trajectory of the Q-score after our simulation.

## Resources
*   **Relevant study:** [Classifying schizophrenic and controls from fMRI data using graph theoretic framework and community detection](https://link.springer.com/article/10.1007/s13721-023-00415-4) *(Note: our project differentiates itself by focusing on topological simulation and specific sub-network breakdowns rather than binary Machine Learning classification).*
* [Preprocessed Dataset](https://data.mendeley.com/datasets/3h4mt7xryk/1)  
* [Original Dataset](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html) 

## Task Distribution

*   **Ana Bog:** Analysis of the results & visualizations (generating the 3D brain maps via `nilearn`, highlighting the bridge nodes, and creating the "healing" Q-score comparison charts).
*   **Yannick Künzli:** Core implementations (calculating the Participation Coefficient, identifying the malicious bridges, and building the node deletion simulation loop).
*   **Yann Gourraud:** Core implementations (running the baseline Leiden algorithm, calculating the initial Modularity Q-Scores, and performing statistical validations between groups).
*   **Mathilde Voyame:** Data processing & result analysis (ingesting the `.nii.gz` files, applying the brain atlas to extract the network matrices, and researching the biological function of our identified nodes). 

*Additionally, everyone will do their own literature research to support their specific tasks.*
