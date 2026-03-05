# SMA-schizo-nets

| Name | Github Handle |
| --- | --- |
| Ana Bog | @kkentia |
| Yannick Künzli | @YannickKunz |
| Yann Gourraud | @Dace23 |
| Mathilde Voyame | @matvoyame |

## Project Description

We are using network graph theory to understand the human brain, which can be treated as a complex and modular IT system. Using preprocessed fMRI time series data from the COBRE dataset, we aim to build weighted network graphs where each node represents a brain region and edges represent functional connectivity between the nodes.

**Biological Problem**

In the normal human brain, these networks function like well-encapsulated microservices. For example, the "Executive Network" (focusing) and the "Default Mode Network" (daydreaming) function independently. However, in Schizophrenia, this network encapsulation breaks down. The boundaries between these networks get blurred, and they start to "cross-wire," causing the brain to misinterpret internal thoughts as external hallucinations.

**Computational & Analytical Approach**

To quantitatively prove this structural breakdown of the Schizophrenia network, our approach will be to:

Run Community Detection: Use the Leiden/Louvain algorithm to run community detection on our brain network graphs.  
Measure Modularity (Q Score): Use this to calculate the Q Score of the network to understand how strictly separated these communities are.  
We expect a statistically significant difference in the Q Score between Schizophrenic and Healthy networks.  

Analysis of Clinical Correlation (Severity): We will compare the network metrics we computed with the patients' phenotypic data. Our aim is to show that mathematically, the fragmentation of the network correlates with the severity of the patients' symptoms.

Visualization of the Breakdown: We will create colored and interactive 3D network graphs. The aim is to visually show how the nodes that normally form a neat community in a normal network are scattered and bottlenecked in a Schizophrenic network.

[Relevant study](https://link.springer.com/article/10.1007/s13721-023-00415-4)  
[Preprocessed Dataset](https://data.mendeley.com/datasets/3h4mt7xryk/1)  
[Original Dataset](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)  

## Task distribution

Ana Bog: analysis of the results & visualizations.  
Yannick Künzli: core implementations (algorithms & validation metrics).  
Yann Gourraud: core implementations (algorithms & validation metrics).  
Mathilde Voyame: data processing & result analysis.  

Additionally, everyone will do their own litterature research. 
