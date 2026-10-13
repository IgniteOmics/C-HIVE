import logging
from collections.abc import Sequence
from typing import Literal, Optional
import torch
import numpy as np
import pandas as pd
from anndata import AnnData
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalJointObsField
from .module import TorchVAE
from .layers import Aggregator
from .trainingplans import AjustTrainingPlan
from .data_splitting import GraphDataSplitter
from scvi.utils import setup_anndata_dsp
from scvi.train import TrainRunner
from scvi.utils._docstrings import devices_dsp
from scvi.model._utils import get_max_epochs_heuristic
from scvi.model.base import BaseModelClass

logger = logging.getLogger(__name__)

class CHive(BaseModelClass):
    """``EXPERIMENTAL`` single-cell Variational Inference :cite:p:`Lopez18`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.JaxSCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_bg_latent
        Dimensionality of the background latent space.
    n_salient_latent
        Dimensionality of the salient latent space.
    dropout_rate
        Dropout rate for neural networks.
    sample_id
        Key in `adata.obs` that contains the sample ID for each cell.
    design
        Key in `adata.obs` that contains the design matrix for each cell. represents whether the cell is from a background or salient data.
    n_layer
        Number of heirarchical layers for the background latent space.
    knn
        Number of nearest neighbors to consider for the graph construction or raidus distance.
    mode
        Mode for the graph construction, either "KNN" or "Radius". If "KNN", the `knn` parameter is used to determine the number of neighbors. If "Radius", the `knn` parameter is used as the radius distance.
    **model_kwargs
        Keyword args for :class:`~ scvi.module.base.BaseModuleClass`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> CHive.model.CHive.setup_anndata(adata, design=['label'], sample_id="dataset")
    >>> model = CHive.model.CHive(adata, sample_id="dataset", design="label")
    >>> model.train()
    >>> adata.obsm[f"X_CHive_BG"] = model.get_latent_representation(layer=-2)
    >>> adata.obsm[f"X_CHive_Salient"] = model.get_latent_representation()
    """
    _module_cls = TorchVAE
    _data_splitter_cls = GraphDataSplitter
    _training_plan_cls = AjustTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_bg_latent: int = 10,
        n_salient_latent: int = 10,
        dropout_rate: float = 0.1,
        sample_id="SampleID",
        design="label",
        n_layer: int = 3,
        knn = 6,
        mode="KNN",
        gene_graph=False,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_batch = self.summary_stats.n_batch
        n_cats_per_cov = self.summary_stats.get("n_extra_continuous_covs", 0)
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_output=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_bg_latent=n_bg_latent,
            n_salient_latent=n_salient_latent,
            dropout_rate=dropout_rate,
            n_layer=n_layer,
            **model_kwargs,
        )
        self.knn = knn
        self.mode = mode
        self.sample_id = sample_id
        self.design = design
        self.gene_graph = gene_graph
        self._model_summary_string = ""
        self.n_salient_latent = n_salient_latent
        self.n_bg_latent = n_bg_latent
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        design: Optional[list] = None,
        sample_id: Optional[str] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, sample_id),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, design),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def _make_data_loader(
        self,
        adata,
        indices = None,
        batch_size= None,
        shuffle: bool = False,
        **data_loader_kwargs,
    ):
        """Create a AnnDataLoader object for data iteration.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        shuffle
            Whether observations are shuffled each iteration though
        data_loader_kwargs
            Kwargs to the class-specific data loader class
        """
        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            raise AssertionError(
                "AnnDataManager not found. Call `self._validate_anndata` prior to calling this "
                "function."
            )

        adata = adata_manager.adata

        if batch_size is None:
            batch_size = settings.batch_size
        if indices is None:
            indices = np.arange(adata.n_obs)

        if "num_workers" not in data_loader_kwargs:
            data_loader_kwargs.update({"num_workers": settings.dl_num_workers})
        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=0,
            validation_size=1.0,
            batch_size=batch_size,
            shuffle_set_split=False,
            sample_id = self.sample_id,
            hop=6,
            dis = self.knn,
            mode=self.mode,
            design=self.design,
            gene_graph=self.gene_graph
        )
        data_splitter.setup()
        dl = data_splitter.val_dataloader()
        return dl

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        layer: int = -1,
        indices: Optional[Sequence[int]] = None,
        return_scale: bool = False,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        agg=False,
    ) -> np.ndarray:
        r"""Return the latent representation for each cell.

        This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        layer
            which layer to use. -1 refers to the salient space and -2 refers to the background space
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        return_scale
            Whether to return the scale of the posterior distribution or a sample. Depreciated
        n_samples
            Number of samples to use for computing the latent representation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        agg
            Whether to spatial smooth the latent representation across cells of the neighbors.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        self._check_if_trained(warn=False)
        aggregator = Aggregator(aggr='mean')
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )
        self.module.eval()
        latent = []
        latent_scale = []
        idx = []
        for array_dict in scdl:
            out = self.module._generic_inference(
                array_dict["x"], array_dict["edge_index"], 
                array_dict["design"], None, n_samples=n_samples
            )
            if not return_scale:
                qz = out["qz"]
                qz = qz[layer]
                z = qz.mean
                if agg:
                    edge_index = array_dict["edge_index"].to(z.device)
                    z = aggregator(z, edge_index)
                z = z[slice(None, array_dict.batch_size)]

            else:
                z = out["qz"].loc
                scale = out["qz"].scale
                latent_scale.append(scale)
            latent.append(z)
            idx.append(array_dict["index"][slice(None, array_dict.batch_size)])
        concat_axis = 0 if ((n_samples == 1) or return_scale) else 1
        latent = torch.concat(latent, axis=concat_axis).detach().cpu()
        idx = torch.concat(idx, axis=0).to(latent.device).long()
        reordered_latent = torch.zeros_like(latent)
        # Reorder latent based on idx
        reordered_latent[idx] = latent

        if return_scale:
            latent_scale = torch.concat(latent_scale, axis=concat_axis)
            return reordered_latent.numpy(), latent_scale.detach().cpu().numpy()

        return reordered_latent.numpy()

    @devices_dsp.dedent
    def train(
        self,
        max_epochs=None,
        accelerator = "auto",
        devices = "auto",
        train_size = 0.9,
        validation_size= None,
        shuffle_set_split= True,
        batch_size= 128,
        early_stopping= False,
        datasplitter_kwargs= None,
        plan_kwargs = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        lr
            Learning rate to use during training.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.JaxTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            max_epochs = max(get_max_epochs_heuristic(self.adata.n_obs), 10)

        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            sample_id = self.sample_id,
            design=self.design,
            hop=6,
            mode=self.mode,
            dis=self.knn,
            gene_graph=self.gene_graph,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, n_epochs_kl_warmup=max_epochs, **plan_kwargs)
        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    def get_loadings(self) -> pd.DataFrame:
        """Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        cols = [f"Topic C{i+1}" for i in range(self.n_bg_latent)] + [f"Topic S{i+1}" for i in range(self.n_salient_latent)]
        var_names = self.adata.var_names
        loadings = pd.DataFrame(self.module.get_loadings(), index=var_names, columns=cols)

        return loadings