from typing import Optional
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import numpy as np
from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from scvi.dataloaders._data_splitting import validate_data_split
from torch_geometric.loader import NeighborLoader
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def fast_knn_faiss(data, k):
    import faiss
    index = faiss.IndexFlatL2(data.shape[1])  # L2 距离
    index.add(data)
    distances, indices = index.search(data, k)
    return distances, indices


def knn_sklearn(data, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors(data)
    return distances, indices


def Spatial_Dis_Cal(
    adata,
    dis=None,
    model="Radius",
    original_indices=None,
    design=None,
    gene=False,
    verbose=True,
):
    """
    Calculate the spatial neighbor networks, as the distance between two spots.
    Parameters
    ----------
    adata: AnnData object of scanpy package.
    rad_dis: radius distance when model='Radius'
    knn_dis: The number of nearest neighbors when model='KNN'
    model: The network construction model.
    """
    assert model in ["Radius", "KNN", "BallTree", "OT", "convex", "delaunay"]
    if verbose:
        print("------Calculating spatial graph...")
    if not gene:
        coor = adata.obsm["spatial"]
    else:
        sc.pp.pca(adata)
        coor = pd.DataFrame(adata.obsm["X_pca"])
    n_spots = coor.shape[0]

    if model == "KNN":
        if n_spots > 100000:
            if verbose:
                print(
                    f"Data has {n_spots} spots, using FAISS for faster KNN computation."
                )
            coor_np = coor.astype("float32")
            distances, indices = fast_knn_faiss(coor_np, dis + 1)
        else:
            if verbose:
                print(f"Data has {n_spots} spots, using sklearn for KNN computation.")
            distances, indices = knn_sklearn(coor,dis + 1)

        row = np.repeat(np.arange(n_spots), dis)
        col = indices[:, 1:].flatten()
        dist = distances[:, 1:].flatten()
        adj_matrix = sp.coo_matrix((dist, (row, col)), shape=(n_spots, n_spots))
        adj_matrix.setdiag(1.0)
        adata.uns["Spatial_Net"] = adj_matrix

    if model == "Radius":
        nbrs = NearestNeighbors(radius=dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

        row = np.repeat(np.arange(n_spots), [len(i) for i in indices])
        col = np.concatenate(indices)
        dist = np.concatenate(distances)
        adj_matrix = sp.coo_matrix((dist, (row, col)), shape=(n_spots, n_spots))
        adj_matrix.setdiag(1.0)

    if verbose:
        aver = adj_matrix.nnz / n_spots
        print(f"Average edges: {aver} .")

    edgeList = adj_matrix.nonzero()
    if isinstance(adata.X, np.ndarray):
        x = torch.FloatTensor(adata.X)
    else:
        x = torch.FloatTensor(adata.X.todense())
    edge_index = torch.LongTensor(np.array([edgeList[0], edgeList[1]]))
    edge_index = to_undirected(edge_index)
    design = torch.Tensor(np.array(adata.obs[design]))
    data = Data(
        edge_index=edge_index,
        x=x,
        design=design,
        index=torch.Tensor(np.array(original_indices))
    )
    return data


def adata2graph(
    adatas, design, dis=4,  gene=False, model="KNN", original_indices=None, verbose=False
):
    data_all = []
    for i, adata in enumerate(adatas):
        adata = adata.copy()
        data = Spatial_Dis_Cal(
            adata,
            dis=dis,
            model=model,
            design=design,
            gene=gene,
            original_indices=original_indices[i],
            verbose=verbose
        )
        data_all.append(data)
    return data_all


def merge_graphs(graph_list):
    node_offset = 0 
    all_x = []
    all_edge_index = []
    all_design = []
    idx_list = []

    for i, graph in enumerate(graph_list):
        num_nodes = graph.x.size(0)
        edge_index = graph.edge_index + node_offset
        all_x.append(graph.x)
        all_edge_index.append(edge_index)
        all_design.append(graph.design)
        idx_list.append(graph.index)
        node_offset += num_nodes 

    x = torch.cat(all_x, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1)
    design = torch.cat(all_design, dim=0)
    idx_vector = torch.cat(idx_list, dim=0)

    data = Data(x=x, edge_index=edge_index, design=design.reshape(-1, 1), 
                index=idx_vector)
    return data



class GraphDataSplitter(DataSplitter):
    """Creates ContrastiveDataLoader for training, validation, and test set.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via
        ``setup_anndata``.
    background_indices: Indices for background samples in adata.
    target_indices: Indices for target samples in adata.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and
        test set are split in the sequential order of the data according to
        `validation_size` and `train_size` percentages.
    load_sparse_tensor
        If `True`, loads sparse CSR or CSC arrays in the input dataset as sparse
        :class:`~torch.Tensor` with the same layout. Can lead to significant
        speedups in transferring data to GPUs, depending on the sparsity of the data.
        Passed into :class:`~scvi.data.AnnDataLoader`.
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. Data loader class is
        :class:`~scvi.dataloaders.AnnDataLoader`.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        pin_memory: bool = False,
        dis=6,
        hop=2,
        sample_id="sample_id",
        design="design",
        mode="KNN",
        gene_graph=False,
        **kwargs,
    ) -> None:
        super().__init__(adata_manager=adata_manager)
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.load_sparse_tensor = load_sparse_tensor
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory
        self.sample_id = sample_id
        self.design = design
        self.gene_graph = gene_graph
        self.dis = dis
        self.mode = mode
        self.graph = self.gen_graph()
        self.hop = hop

        if self.train_size != 0:
            self.n_train, self.n_val = validate_data_split(
                self.graph.num_nodes, self.train_size, self.validation_size
            )
        else:
            self.n_val = self.graph.num_nodes
            self.n_train = 0


    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.graph.num_nodes)

        if self.shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        self.val_idx = indices[: n_val]
        self.train_idx = indices[n_val : (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train):]

    def gen_graph(self):
        adata = self.adata_manager.adata

        if self.gene_graph:
            adatas = [adata]
            original_indices = [list(range(adata.shape[0]))]
        else:
            adatas = []
            original_indices = []
            samples = pd.unique(adata.obs[self.sample_id])
            for sample in samples:
                adatas.append(adata[adata.obs[self.sample_id] == sample])
                original_indices.append(np.where(adata.obs[self.sample_id] == sample)[0])
        graphs = adata2graph(
            adatas,
            dis=self.dis,
            design=self.design,
            gene=self.gene_graph,
            model=self.mode,
            original_indices=original_indices,
            verbose=False
        )
        combined_graph = merge_graphs(graphs)
        return combined_graph

    def train_dataloader(self):
        loader = NeighborLoader(
            data=self.graph,
            num_neighbors=[-1]*self.hop,
            shuffle=True,
            **self.data_loader_kwargs
        )

        return loader

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            loader = NeighborLoader(
                data=self.graph,
                num_neighbors=[-1]*self.hop,
                shuffle=False,
                **self.data_loader_kwargs
            )

            return loader
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            loader = NeighborLoader(
                data=self.graph,
                num_neighbors=[-1]*self.hop,
                shuffle=False,
                **self.data_loader_kwargs
            )

            return loader
        else:
            pass
