import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing

class Aggregator(MessagePassing):
    def __init__(self, aggr='mean'):  # 'mean', 'add', 'max'
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        # x: (num_nodes, feature_dim)
        # edge_index: (2, num_edges)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  

    def update(self, aggr_out):
        return aggr_out  


class GCNBlock(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout_rate, 
                 normalisation="layernorm", end_dense=False, use_residual=False):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout_rate = dropout_rate
        self.normalisation = normalisation
        self.end_dense = end_dense
        self.use_residual = use_residual

        # GCN layer
        self.dense = GATConv(self.n_input, self.n_hidden)

        # Residual connection adjustment layer (only used if n_input != n_hidden)
        if self.use_residual and self.n_input != self.n_hidden:
            self.residual_layer = nn.Linear(self.n_input, self.n_hidden, bias=False)
        else:
            self.residual_layer = None

        # Normalisation layer
        if self.normalisation == "layernorm":
            self.norm = nn.LayerNorm(self.n_hidden, eps=1e-6)
        else:
            self.norm = nn.BatchNorm1d(self.n_hidden, momentum=0.1)
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # Optional final dense layer
        if self.end_dense:
            self.dense1 = GATConv(self.n_hidden, self.n_output)

    def forward(self, x, edge_index):
        residual = x

        x = self.dense(x, edge_index)
        x = self.dropout(x)
        x = self.norm(x)
        x = F.relu(x)

        # Add residual connection if enabled
        if self.use_residual:
            if self.residual_layer is not None:
                residual = self.residual_layer(residual)
            x = x + residual

        if self.end_dense:
            x = self.dense1(x, edge_index)

        return x


class MLPLayer(nn.Module):
    """A simple MLP layer"""
    def __init__(self, n_input, n_hidden, n_output, dropout_rate, 
                 normalisation="layernorm", end_dense=False):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout_rate = dropout_rate
        self.normalisation = normalisation
        self.end_dense = end_dense

        # Dense layer
        self.dense = nn.Linear(self.n_input, self.n_hidden)

        # Normalisation layer
        if self.normalisation == "layernorm":
            self.norm = nn.LayerNorm(self.n_hidden, eps=1e-6)
        else:
            self.norm = nn.BatchNorm1d(self.n_hidden, momentum=0.1)

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # Optional second dense layer
        if self.end_dense:
            self.dense1 = nn.Linear(self.n_hidden, self.n_output)

    def forward(self, x, edge_index=None):
        # Apply the first dense layer
        x = self.dense(x)

        # Apply dropout
        x = self.dropout(x)

        # Apply normalisation
        x = self.norm(x)

        # Apply ReLU activation
        x = F.relu(x)

        # Apply second dense layer if end_dense is True
        if self.end_dense:
            x = self.dense1(x)

        return x


class BottomUpLayer(nn.Module):
    """Returns changes to the posterior distributions"""

    def __init__(self, n_input, n_hidden, n_output, dropout_rate, n_layer=1):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout_rate = dropout_rate
        self.n_layer = n_layer

        self.layers = nn.ModuleList([
            MLPLayer(
                n_input=self.n_input,
                n_hidden=self.n_hidden,
                n_output=self.n_output,
                dropout_rate=self.dropout_rate,
                end_dense=False,
            )
            for _ in range(self.n_layer)
        ])

    def forward(self, x, edge_index):
        ys = []
        for i in range(self.n_layer):
            x = self.layers[i](x, edge_index)
            ys.append(x)
        return ys



class TopDownLayer(nn.Module):
    """Generative path. Takes in posterior from
    previous layer and outputs next layer prior and posterior"""

    def __init__(self, prior_input, posterior_input, design_input, 
                 n_hidden, n_output, dropout_rate):
        super().__init__()
        self.prior_input = prior_input
        self.posterior_input = posterior_input
        self.design_input = design_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout_rate = dropout_rate

        # Define layers
        self.prior_layer = GCNBlock(
            n_input=self.prior_input,
            n_hidden=self.n_hidden,
            n_output=self.n_output * 2,
            dropout_rate=self.dropout_rate,
            end_dense=True
        )
        self.dense = MLPLayer(
            n_input=self.prior_input,
            n_hidden=self.n_hidden,
            n_output=self.n_hidden,
            dropout_rate=self.dropout_rate
        )
        self.posterior_layer = nn.Linear(self.posterior_input, self.n_output * 2)
        self.prior_design_layer = nn.Linear(self.design_input, self.n_output, bias=False)
        self.posterior_design_layer = nn.Linear(self.design_input, self.n_output)

    def forward(self, z, edge_index, design, bottom_up, prev_vals, agg=False):
        if z is not None:
            pz_vals = self.prior_layer(z, edge_index)
            if prev_vals != None:
                pz_vals = pz_vals+prev_vals
            pz_mean, pz_log_var = torch.split(pz_vals, pz_vals.size(1) // 2, dim=1)
            pz_log_var = torch.clamp(pz_log_var, min=-7, max=7)
            pz_std = torch.sqrt(torch.exp(pz_log_var))
            # pz_mean = F.softplus(pz_mean)

            if design is not None:
                pz_mean = pz_mean * self.prior_design_layer(design)
                pz_std = pz_std * torch.unsqueeze((design.sum(dim=1) >= 1), -1)
                pz_std = pz_std + (torch.unsqueeze((design.sum(dim=1) <= 1), -1) * 5e-2)
            pz = dist.Normal(pz_mean, pz_std)
        else:
            pz_mean = torch.tensor(0.0)  # mean of the distribution
            pz_std = torch.tensor(1.0)   # standard deviation of the distribution
            pz_vals = torch.tensor(0.0)
            pz = dist.Normal(pz_mean, pz_std)

        if z is not None:
            z = self.dense(z, edge_index)
            qz_vals = self.posterior_layer(torch.cat([bottom_up, z], dim=1))
        else:
            qz_vals = self.posterior_layer(bottom_up)
        qz_mean, qz_log_var = torch.split(qz_vals, qz_vals.size(1) // 2, dim=1)
        qz_log_var = torch.clamp(qz_log_var, min=-7, max=7)
        qz_std = torch.sqrt(torch.exp(qz_log_var))


        if design is not None:
            posterior_design = self.posterior_design_layer(design)
            qz_mean = qz_mean * (posterior_design)
        qz = dist.Normal(qz_mean, qz_std)
        z = qz.rsample()

        return z, pz, qz, pz_vals

