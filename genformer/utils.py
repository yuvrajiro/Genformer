import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Optional
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from transformers import set_seed
except ImportError:
    set_seed = None

try:
    from datasets import disable_progress_bar
except ImportError:
    disable_progress_bar = None
import logging

def setup_logger():
    # Lightning 2.x
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    # Older Lightning versions
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    
    class IgnoreDartsIndexError(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "Integer index out of range" not in msg

    logger = logging.getLogger("darts.timeseries")
    logger.addFilter(IgnoreDartsIndexError())

class Deterministic:
    def __init__(self):
        pass

    def init_all(self, seed=0, disable_list=['cuda_block']):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if 'cuda_block' not in disable_list: # stuck when train deberta
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if 'torch_deter_algo' not in disable_list: # consumn more gpu sometimes
            torch.use_deterministic_algorithms(True, warn_only=True)
        if set_seed is not None:
            set_seed(seed)
        if cv2 is not None:
            cv2.setRNGSeed(seed)
        if disable_progress_bar is not None:
            disable_progress_bar()

deterministic = Deterministic()


class GraphConv(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        edges,
        num_nodes,
        aggregation_type: str = "mean",
        combination_type: str = "add",
        activation: Optional[str] = None
    ):
        super(GraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.register_buffer("src_nodes", torch.tensor(edges[0], dtype=torch.long))
        self.register_buffer("dst_nodes", torch.tensor(edges[1], dtype=torch.long))
        self.num_nodes = num_nodes
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        # weight parameter
        self.weight_self = nn.Parameter(torch.empty(in_feat, out_feat))
        self.weight_neigh = nn.Parameter(torch.empty(in_feat, out_feat))
        # Xavier Glorot initialization
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.xavier_uniform_(self.weight_neigh)

        self.norm = nn.LayerNorm(out_feat)

        # Activation
        if activation is None:
            self.activation = lambda x: x
        elif isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def aggregate(self, neighbour_representations: torch.Tensor):
        src_nodes = self.src_nodes
        dst_nodes = self.dst_nodes
        num_nodes = self.num_nodes

        if self.aggregation_type == "sum":
            aggregated = torch.zeros(
                (num_nodes,) + neighbour_representations.shape[1:],
                device=neighbour_representations.device
            )
            aggregated.index_add_(0, src_nodes, neighbour_representations)

        elif self.aggregation_type == "mean":
            aggregated = torch.zeros(
                (num_nodes,) + neighbour_representations.shape[1:],
                device=neighbour_representations.device
            )
            counts = torch.zeros(num_nodes, device=neighbour_representations.device).float()
            aggregated.index_add_(0, src_nodes, neighbour_representations)
            counts.index_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float))
            counts = counts.clamp(min=1).view(-1, *([1]*(neighbour_representations.dim()-1)))
            aggregated = aggregated / counts

        elif self.aggregation_type == "max":
            # Group-wise max per node
            aggregated = torch.full(
                (num_nodes,) + neighbour_representations.shape[1:],
                float('-inf'), device=neighbour_representations.device
            )
            for i in range(src_nodes.shape[0]):
                aggregated[src_nodes[i]] = torch.max(
                    aggregated[src_nodes[i]], neighbour_representations[i]
                )

        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

        return aggregated

    def compute_nodes_representation(self, features: torch.Tensor):
        """
        features: (num_nodes, batch_size, input_seq_len, in_feat)
        returns: (num_nodes, batch_size, input_seq_len, out_feat)
        """
        return torch.matmul(features, self.weight_self)  # last-dim matmul

    def compute_aggregated_messages(self, features: torch.Tensor):
        dst_nodes = self.dst_nodes
        # gather neighbors
        neighbour_representations = features[dst_nodes]
        aggregated_messages = self.aggregate(neighbour_representations)
        return torch.matmul(aggregated_messages, self.weight_neigh)

    def update(self, nodes_representation: torch.Tensor, aggregated_messages: torch.Tensor):
        if self.combination_type == "concat":
            h = torch.cat([nodes_representation, aggregated_messages], dim=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")
        h = self.activation(h)
        self.norm = self.norm.to(h.device)
        h = self.norm(h)
        return h

    def forward(self, features: torch.Tensor):
        """
        features: (num_nodes, batch_size, input_seq_len, in_feat)
        returns:  (num_nodes, batch_size, input_seq_len, out_feat)
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)


def energy_score_loss_st(y_true, y_pred_samples):
    """
    Calculates the Energy Score loss for spatiotemporal targets.
    Args:
        y_pred_samples (torch.Tensor): Shape (M, B, T_out, N, D), M = num_samples
        y_true (torch.Tensor): Shape (B, T_out, N, D)
    """
    M, B, T, N, D = y_pred_samples.shape
    y_pred_samples = y_pred_samples.permute(1, 0, 2, 3, 4) # (B, M, T, N, D)
    y_true_expanded = y_true.unsqueeze(1) # (B, 1, T, N, D)

    # First term: E||Y_hat - y||
    term1 = torch.norm(y_pred_samples - y_true_expanded, p=2, dim=(-3, -2, -1)).mean(dim=1)

    # Second term: 0.5 * E||Y_hat - Y_hat'||
    y_pred_flat = y_pred_samples.reshape(B, M, -1) # -> (B, M, T*N*D)

    # Calculate pairwise distances
    diff = y_pred_flat.unsqueeze(2) - y_pred_flat.unsqueeze(1)   # (B, M, M, dim)
    eps = torch.tensor(1e-8, device=diff.device)
    dist_matrix = torch.sqrt((diff ** 2).sum(-1) + eps)
    term2 = 0.5 * dist_matrix.mean(dim=(1, 2))

    return (term1 - term2).mean()


def calibration_loss(future_target, samples, alpha, temperature, width_penalty):
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower = torch.quantile(samples, lower_q, dim=0)
    upper = torch.quantile(samples, upper_q, dim=0)
    
    # soft coverage
    soft_coverage = torch.sigmoid(temperature * (future_target - lower)) * torch.sigmoid(temperature * (upper - future_target))
    soft_coverage = torch.clamp(soft_coverage, min=1e-6, max=1-1e-6).mean()

    # interval width regularization
    width = (upper - lower).abs().mean()
    calibration_loss = (soft_coverage - alpha) ** 2 - width_penalty * width
    coverage = (((future_target >= lower) & (future_target <= upper)).float().mean(dim=(0, 2, 3)).mean())

    return calibration_loss, coverage


def get_values_safe(ts_or_tensor):
    if hasattr(ts_or_tensor, "all_values"):
        # Darts TimeSeries
        return ts_or_tensor.all_values()
    else:
        # Already a tensor or numpy array
        return ts_or_tensor


def generate_forecasts(model, history, m_samples, past_covs=None, fut_covs=None, static_covs=None, unstandardize=None, device="cpu"):
    net = model.model.to(device) if hasattr(model, 'model') else model.to(device)
    net.eval()

    # Get the 2D values: (Time, Total_Features)
    h_values = get_values_safe(history)     # tensor of shape (T, N, D)
    
    if isinstance(h_values, np.ndarray):
        h_values = torch.from_numpy(h_values).float()
    elif not isinstance(h_values, torch.Tensor):
        h_values = torch.as_tensor(h_values, dtype=torch.float32)

    if h_values.ndim == 2:                  # D missing
        h_values = h_values.unsqueeze(-1)   # add D=1

    # Correct Reshaping logic: split Total_Features into Nodes and Node_Features
    T_len, N, D = h_values.shape
    # Create (1, T, N, D)
    history_tensor = h_values.view(1, T_len, N, D)
    history_tensor = history_tensor.to(device)

    forecasts = []
    with torch.no_grad():
        for _ in range(m_samples):
            x = history_tensor # 4D: (1, T, N, D)
            B, T, N, D = x.shape

            # Spatial processing (GCN)
            x = x.permute(2, 0, 1, 3) # (N, B, T, D)
            x = net.gcn(x)
            x = x.permute(1, 2, 0, 3) # (B, T, N, Gcn_Out)

            # Temporal processing: flatten spatial dim into features for the Transformer
            x = x.reshape(B, T, N * net.gcn_out_feat)
            x_exp = (x.float(), past_covs, fut_covs)
            y = net(x_exp)

            # Reshape output back to spatial dimensions (B, T_out, N, D_out(gcn_out_feat))
            y = y.view(B, -1, N, net.gcn_out_feat)
            forecasts.append(y.squeeze(0)) # Remove batch dim for stacking

    # Stack to get (m_samples, T_out, N, D_out)
    forecasts = torch.stack(forecasts, dim=0)
    forecasts = forecasts.to(device)

    # Unstandardize if provided
    if unstandardize is not None:
        mean, std = unstandardize
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(std, dtype=torch.float32, device=device)

        # Reshape mean/std to (1, 1, N, D) for broadcasting
        mean = mean.view(1, 1, N, -1)
        std = std.view(1, 1, N, -1)
        forecasts = forecasts * std + mean

    return forecasts
