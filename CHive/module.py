import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.distributions import Normal
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from .layers import BottomUpLayer, TopDownLayer
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling



class PyTorchEncoder(nn.Module):
    """Encoder for PyTorch VAE. Inference -> outputs changes to posterior from prior"""

    def __init__(self, n_input, n_design, n_bg_latent, 
                 n_salient_latent, n_hidden, n_layer, dropout_rate):
        super(PyTorchEncoder, self).__init__()
        self.n_input = n_input
        self.n_bg_latent = n_bg_latent
        self.n_salient_latent = n_salient_latent
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.n_design = n_design
        # Bottom-up layers for background
        self.bottom_up_layer_bg = nn.ModuleList([
            BottomUpLayer(self.n_input, self.n_hidden, self.n_hidden, 
                               dropout_rate=self.dropout_rate, n_layer=1)
            for i in range(self.n_layer)
        ])

        # Top-down layers for background
        self.top_down_layer_bg = nn.ModuleList([
            TopDownLayer(self.n_bg_latent, self.n_hidden, self.n_design, 
                         self.n_hidden, self.n_bg_latent, dropout_rate=self.dropout_rate) if i == 0
            else TopDownLayer(self.n_bg_latent, self.n_hidden + self.n_hidden, self.n_design,
                              self.n_hidden, self.n_bg_latent, dropout_rate=self.dropout_rate)
            for i in range(self.n_layer)
        ])

        # Bottom-up layer for salient latent
        self.bottom_up_layer_salient = BottomUpLayer(self.n_input, self.n_hidden, self.n_hidden, 
                                                     dropout_rate=self.dropout_rate, n_layer=1)

        # Top-down layer for salient latent
        self.top_down_layer_salient = TopDownLayer(self.n_bg_latent, self.n_hidden + self.n_hidden, self.n_design,
                                                   self.n_hidden, self.n_salient_latent, dropout_rate=self.dropout_rate)
        self.bg_linear = nn.ModuleList([
            nn.Linear(self.n_bg_latent, self.n_bg_latent)
            for i in range(self.n_layer)
        ])

    def forward(self, x, edge_index, design, batch_size=None):
        """Forward pass"""
        # Log transform input
        # x = torch.log1p(x)
        pz = []
        qz = []

        z_bg = None
        pz_bg_vals = None
        pz_salient_vals = None

        # Bottom-up and top-down path for background latent 
        for i in range(self.n_layer):
            bottom_up = self.bottom_up_layer_bg[i](x, edge_index)
            z_bg, pz_bg, qz_bg, pz_bg_vals = self.top_down_layer_bg[i](z_bg, edge_index, None, bottom_up[0], pz_bg_vals)
            if i != 0:
                pz_bg = self.slice_distribution(pz_bg, batch_size)
            qz_bg = self.slice_distribution(qz_bg, batch_size)
            pz.append(pz_bg)
            qz.append(qz_bg)

        # Bottom-up for salient latent 
        bottom_up_salient = self.bottom_up_layer_salient(x, edge_index)

        # Top-down for salient latent 
        z_salient, pz_salient, qz_salient, pz_salient_vals = self.top_down_layer_salient(
            z_bg, edge_index, design, bottom_up_salient[0], pz_salient_vals
        )
        if batch_size is not None:
            pz_salient = self.slice_distribution(pz_salient, batch_size)
            qz_salient = self.slice_distribution(qz_salient, batch_size)
            z_bg = z_bg[slice(None, batch_size)]
            z_salient = z_salient[slice(None, batch_size)]
        pz.append(pz_salient)
        qz.append(qz_salient)


        return z_bg, z_salient, pz, qz
    
    def slice_distribution(self, pz, batch_size):
        pz_mean = pz.loc[:batch_size]
        pz_std = pz.scale[:batch_size]

        pz_batch = Normal(pz_mean, pz_std)
        return pz_batch




class PyTorchDecoder(nn.Module):
    """Decoder for PyTorch VAE."""

    def __init__(self, n_input, n_batch, n_output, dropout_rate, n_hidden):
        super(PyTorchDecoder, self).__init__()
        self.n_input = n_input
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_output = n_output

        # Define the layers
        self.dense1 = nn.Linear(self.n_input, self.n_hidden)
        self.dense2 = nn.Linear(self.n_batch, self.n_hidden)
        self.dense3 = nn.Linear(self.n_hidden, self.n_output)

        # Layer normalization
        self.layernorm = nn.LayerNorm(self.n_hidden, eps=1e-6)

        # Dispersion parameter (learnable parameter initialized with random normal values)
        self.disp = nn.Parameter(torch.randn(self.n_output, ))

    def forward(self, z, batch):
        """Forward pass"""
        # training = training if training is not None else self.training

        h = self.dense1(z)
        batch_output = self.dense2(batch)
        h += batch_output
        h = self.layernorm(h)
        h = F.relu(h)
        h = self.dense3(h)

        return h, self.disp

    

class LinearDecoder(nn.Module):
    """Decoder for PyTorch VAE."""

    def __init__(self, n_input, n_batch, n_output, dropout_rate, n_hidden, bias=True):
        super(LinearDecoder, self).__init__()
        self.n_input = n_input
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_output = n_output

        self.dense_batch = nn.Linear(self.n_batch, self.n_output, bias=bias)
        self.dense_emb = nn.Linear(self.n_input, self.n_output, bias=bias)

        # Layer normalization
        self.norm = nn.BatchNorm1d(self.n_output, momentum=0.01, eps=0.001)
        self.disp = nn.Parameter(torch.ones(self.n_output, ))

    def forward(self, z, batch):
        """Forward pass"""
        h_emb = self.dense_emb(z)
        h = h_emb


        return h, self.disp


class TorchVAE(BaseModuleClass):
    """Variational autoencoder model using PyTorch."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_batch: int,
        n_design=1,
        n_cats_per_cov: Optional[list] = None,
        n_hidden: int = 128,
        n_bg_latent: int = 10,
        n_salient_latent: int = 10,
        dropout_rate: float = 0.1,
        n_layer: int = 1,
        gene_likelihood: str = "nb",
        eps: float = 1e-8,
        warmup=True
    ):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_batch = n_batch
        self.n_cats_per_cov = n_cats_per_cov
        self.n_hidden = n_hidden
        self.n_bg_latent = n_bg_latent
        self.n_salient_latent = n_salient_latent
        self.dropout_rate = dropout_rate
        self.n_layer = n_layer
        self.gene_likelihood = gene_likelihood
        self.eps = eps
        self.n_design = n_design
        self.tag = 0
        self.warmup=warmup

        # Initialize encoder and decoder
        self.encoder = PyTorchEncoder(
            n_input=self.n_input,
            n_design=self.n_design,
            n_bg_latent=self.n_bg_latent,
            n_salient_latent=self.n_salient_latent,
            n_hidden=self.n_hidden,
            n_layer=self.n_layer,
            dropout_rate=self.dropout_rate,
        )

        self.decoder = PyTorchDecoder(
            n_input=self.n_bg_latent+self.n_salient_latent,
            n_output=self.n_output,
            n_batch=self.n_batch,
            dropout_rate=0.0,  # No dropout in the decoder
            n_hidden=self.n_hidden,
        )
        self.edge_decoder = InnerProductDecoder()

    def frozen_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.tag = 1

    def change_lineardecoder(self):
        self.decoder = LinearDecoder(
            n_input=self.n_bg_latent+self.n_salient_latent,
            n_batch=self.n_batch,
            n_output=self.n_output,
            dropout_rate=0.0,  # No dropout in the decoder
            n_hidden=self.n_hidden,
        )


    def _get_inference_input(self, tensors: dict[str, torch.Tensor], **kwargs):
        """Parse tensors dictionary for inference related values."""
        x = tensors["x"]
        
        design = tensors["design"]
        edge_index = tensors["edge_index"]
        batch_size = tensors.batch_size
        input_dict = {"x": x, "design": design, "edge_index": edge_index, "batch_size": batch_size}
        return input_dict

    
    @auto_move_data
    def inference(
        self,
        x, edge_index, design, batch_size=None, n_samples= 1
    ):
        z_bg, z_salient, pz, qz = self.encoder(x, edge_index, design, batch_size)
        z = torch.cat([z_bg, z_salient], dim=1)
        return {"qz": qz, "pz": pz, "z": z}


    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        **kwargs,
    ):
        """Parse tensors dictionary for generative related values."""
        """Get input for generative model."""
        batch_size = tensors.batch_size
        x = tensors["x"][slice(None, batch_size)]
        z = inference_outputs["z"][slice(None, batch_size)]
        batch_index = torch.zeros((batch_size,1), dtype=torch.long).to(x.device)
        design = tensors["design"]
        input_dict = {
            "x": x,
            "z": z,
            "batch_index": batch_index,
            "design": design,
        }
        return input_dict


    @auto_move_data
    def generative(self, x, z, batch_index, design):
        """Generative model step."""
        batch = F.one_hot(batch_index, num_classes=self.n_batch).squeeze(-2).to(dtype=torch.float)
        rho, disp = self.decoder(z, batch)
        px = Normal(rho, torch.sqrt(torch.exp(disp)))
        return {"px": px, "rho": rho}
    
    def recon_loss(self, z, pos_edge_index, design=None, neg_edge_index=None):
        if design is None:
            z_tmp = z
        else:
            design = design.squeeze()
            z_tmp = z.clone()
            z_tmp[design == 0, -10:] = 0
        pos_loss = -torch.log(
            self.edge_decoder(z_tmp, pos_edge_index, sigmoid=True) + 1e-15).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z_tmp.size(0))
        neg_loss = -torch.log(1 -
                              self.edge_decoder(z_tmp, neg_edge_index, sigmoid=True) +
                              1e-15).mean()

        return pos_loss + neg_loss

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Compute loss."""
        x = tensors["x"][slice(None, tensors.batch_size)]
        # x = tensors["x"]
        px = generative_outputs["px"]
        pz = inference_outputs["pz"]
        qz = inference_outputs["qz"]

        # Reconstruction los
        reconst_loss = nn.MSELoss(reduction='none')(x, generative_outputs["rho"]).sum(dim=-1)
        # KL Divergence between qz and pz (Prior vs Posterior)
        if self.tag == 1:
            loss = torch.mean(reconst_loss)
            
            return LossOutput(
                loss=loss, reconstruction_loss=reconst_loss
            )
        else:
            if not self.warmup:
                kl_weight = 1

            kl_divergence_z = 0
            for i in range(len(pz)):
                tmp = torch.distributions.kl_divergence(qz[i], pz[i])
                tmp = tmp.sum(-1)
                kl_divergence_z += tmp


            kl_local_for_warmup = kl_divergence_z * kl_weight
            weighted_kl_local = kl_local_for_warmup
            loss = torch.mean(reconst_loss + weighted_kl_local)
            kl_local = kl_divergence_z
            return LossOutput(
                loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local,
            
            )
    
    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        design: torch.Tensor,
        batch_size,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        z_bg, z_salient, pz, qz = self.encoder(x, edge_index, design, batch_size)
        if n_samples > 1:
            z_bg = self._reshape_tensor_for_samples(z_bg, n_samples)
            z_salient = self._reshape_tensor_for_samples(z_salient, n_samples)

        outputs = {
            "z_bg": z_bg,
            "z_salient": z_salient,
            "pz": pz,
            "qz": qz,
        }
        return outputs

            
    @torch.inference_mode()
    def get_loadings(self):
        """Extract per-gene weights in the linear decoder."""
       
        w = self.decoder.dense_emb.weight
        loadings = w.detach().cpu().numpy()
        return loadings
    
    @torch.inference_mode()
    @auto_move_data
    def get_cell_by_topic(self, x,  edge_index=None, design=None, batch_size=None):
        z_bg, z_salient, pz, qz = self.encoder(x, edge_index, design, batch_size)
        s_loc = qz[-1].loc
        z_loc = qz[-2].loc
        return z_loc, s_loc
