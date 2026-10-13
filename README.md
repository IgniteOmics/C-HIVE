# C-HIVE: comparative analysis for spatial or single cell data
## Overview
C-HIVE, a graph-guided hierarchical variational autoencoder designed to disentangle variation features enriched in a target dataset from those shared with a background dataset. By modeling salient embeddings as dependent on shared embeddings, C-HIVE facilitates more accurate identification of phenotype-associated cells in comparative analyses, such as normal versus cancer or mutant versus wild-type conditions. To incorporate spatial or gene expression graph information, we construct an adjacency matrix based on spatial proximity and integrate it into the model using GNNs, enhancing the structural coherence of the latent space. To improve the quality of common and salient representations, we extend C-HIVE into a deeper hierarchical structure, which enables multi-layer modeling of common features.  A linear decoder combined with varimax rotation further promotes structured sparsity, enhancing both robustness and interpretability.

## Installation
We tested our code on a server running Ubuntu 18.04.5 LTS, equipped with 4 NVIDIA A6000 GPUs.
```
git clone https://github.com/JinmiaoChenLab/C-HIVE.git
cd C-HIVE
conda create -n C-HIVE python=3.10.15
conda activate C-HIVE
pip install -r requirements.txt

# install torch
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# install torch_geometrics
pip install torch_geometric==2.6.1 pyg_lib==0.3.1+pt21cu118 torch_scatter==2.1.2+pt21cu118 torch_sparse==0.6.18+pt21cu118 torch_cluster==1.6.2+pt21cu118 torch_spline_conv==1.2.2+pt21cu118 -f https://data.pyg.org/whl/torch-2.1.1+cu121.html

python setup.py install
```

## Tutorial
<!-- * [`spatial data`](./docs/tutorials/Spatial_simulation.ipynb) 
* [`single cell`](./docs/tutorials/single_cell_marfland.ipynb)-->

We provided detailed tutorials on applying C-HIVE to various data. Please refer to [https://C-HIVE.readthedocs.io/en/latest/](https://C-HIVE.readthedocs.io/en/latest/).

## Data
Source of public datasets:
1. HCC dataset: [`Cosmx`](https://www.biorxiv.org/content/10.1101/2024.05.09.593073v1) 
2. Kidney dataset: [`10x Genomics Xenium`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE286051)
3. AD dataset: [`10x Genomics Xenium`](https://www.10xgenomics.com/cn/datasets/xenium-in-situ-analysis-of-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard)
4. Drug dataset: [`Mix-seq`](https://figshare.com/s/139f64b495dea9d88c70)

We have compiled the simulation, in-house, and public datasets into h5ad files. Please refer to [google driver](https://drive.google.com/drive/folders/1hZLNyF0gcPgcBim2M7R8cT_9TfQzx5Rv?usp=drive_link). 