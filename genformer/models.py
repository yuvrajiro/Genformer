import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Optional
import numpy as np
from darts.models.forecasting.transformer_model import TransformerModel, _TransformerModule
from genformer.metrics import energy_score_loss
from genformer.utils import GraphConv, energy_score_loss_st, calibration_loss, generate_forecasts
from genformer.noise import GaussianNoise, UniformNoise


class EnformerModule(_TransformerModule):
    def __init__(self, input_size, output_size, 
            nr_params, *args, num_samples_engression=10, noise_dist="gaussian", noise_std=0.1, **kwargs):
        super().__init__(input_size=input_size, 
            output_size=output_size, 
            nr_params=nr_params, *args, **kwargs)
        
        self.M = num_samples_engression
        self.noise_std = noise_std
        self.noise_dist = noise_dist
        
        if self.noise_dist == "gaussian":
            self.encoder = nn.Sequential(
                GaussianNoise(self.noise_std, 42),
                self.encoder
            )
        elif self.noise_dist == "uniform":
            self.encoder = nn.Sequential(
                UniformNoise(self.noise_std, 42),
                self.encoder
            )
        else:
            raise ValueError("noise_dist must be either `gaussian` or `uniform`.")

    def forward(self, x_in: tuple, *args, **kwargs):
        x_in_float = tuple(
            t.float() if isinstance(t, torch.Tensor) and torch.is_floating_point(t) else t 
            for t in x_in
        )
        return super().forward(x_in_float, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        past_target = batch[0]
        past_covs = batch[1]
        static_covs = batch[4]
        future_target = batch[-1]
        
        batch_size = future_target.size(0)

        past_target_m = past_target.repeat_interleave(self.M, dim=0)
        
        if past_covs is not None:
            past_covs_m = past_covs.repeat_interleave(self.M, dim=0)
            data_m = torch.cat([past_target_m, past_covs_m], dim=2)
        else:
            data_m = past_target_m

        fut_covs_m = batch[3].repeat_interleave(self.M, dim=0) if batch[3] is not None else None
        static_covs_m = static_covs.repeat_interleave(self.M, dim=0) if static_covs is not None else None
        data_m = data_m.float()

        x_expanded = (data_m, fut_covs_m, static_covs_m)

        y_hat_raw = self(x_expanded) 
        
        samples = y_hat_raw.view(batch_size, self.M, y_hat_raw.shape[1], y_hat_raw.shape[2])
        samples = samples.permute(1, 0, 2, 3) # (M, B, T, D)
        
        loss = energy_score_loss(samples, future_target)
        
        self.log("energy_score_train_loss", loss, prog_bar=True, on_epoch=True)
        return loss


class Enformer(TransformerModel):
    r"""
    Enformer: A Deep Generative Transformer for Probabilistic Time Series Forecasting.
    
    This model integrates the engression (distributional regression) principle with 
    a sequence-to-sequence Transformer architecture. Instead of producing point 
    predictions or relying on restrictive parametric likelihoods, Enformer 
    directly estimates the conditional predictive distribution of future observations 
    via noise-driven sampling.

    The architecture explicitly expands the input look-back sequence into an ensemble 
    of `M` replicas, injecting independent stochastic noise into each before processing 
    through the Transformer backbone. The training optimizes a strictly proper scoring 
    rule known as the Energy Score (ES) loss.

    Args:
        input_chunk_length (int): The length of the historical look-back window (\( p \)).
        output_chunk_length (int): The prediction horizon (\( q \)).
        num_samples_engression (int, optional): The number of in-sample forecast 
            trajectories (\( M \)) to generate for the ensemble. Defaults to 10.
        noise_dist (str, optional): The type of noise to inject ('gaussian' or 'uniform').
            Defaults to 'gaussian'.
        noise_std (float, optional): The standard deviation/scale (\( \sigma \)) of 
            the injected noise. Defaults to 0.1.
        random_state (int, optional): Seed for reproducibility. Defaults to 23.
        save_checkpoints (bool, optional): Whether to save PyTorch Lightning checkpoints.
            Defaults to False.
        **kwargs: Additional parameters passed to the underlying `TransformerModel` 
            from Darts (e.g., `n_heads`, `d_model`, `num_encoder_layers`).
    """
    def __init__(self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: typing.Union[str, torch.nn.Module, None] = None,
        custom_encoder: typing.Optional[torch.nn.Module] = None,
        custom_decoder: typing.Optional[torch.nn.Module] = None,
        num_samples_engression: int = 10,
        noise_dist: str = "gaussian",
        noise_std: float = 0.1,
        random_state: typing.Optional[int] = 23,
        save_checkpoints: bool = False,
        **kwargs
    ):
        if num_samples_engression <= 0:
            raise ValueError("num_samples_engression must be positive.")
        if noise_dist not in ["gaussian", "uniform"]:
            raise ValueError("noise_dist must be either `gaussian` or `uniform`.")


        self.num_samples_engression = num_samples_engression
        self.noise_std = noise_std
        self.noise_dist = noise_dist

        # Lightning Trainer arguments
        pl_kwargs = {
            "deterministic": True,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        
        # Determine whether to allow lightning to handle epoch checkpoints (or just run cleanly)
        if save_checkpoints:
            pl_kwargs["enable_checkpointing"] = True
            pl_kwargs["default_root_dir"] = "checkpoints"
        else:
            pl_kwargs["enable_checkpointing"] = False

        super().__init__(input_chunk_length=input_chunk_length,
                         output_chunk_length=output_chunk_length,
                         output_chunk_shift=output_chunk_shift,
                         random_state = random_state,
                         pl_trainer_kwargs=pl_kwargs,
                         **kwargs)
        
        self.model_params.pop('num_samples_engression', None)
        self.model_params.pop('random_state', None)
        self.model_params.pop('noise_std', None)
        self.model_params.pop('noise_dist', None)
        self.model_params.pop('save_checkpoints', None)
        self.model_params.pop('batch_size', None)
        self.model_params.pop('n_epochs', None)
        self.model_params.pop('optimizer_kwargs', None)
        self.model_params.pop('lr_scheduler_cls', None)
        self.model_params.pop('lr_scheduler_kwargs', None)


    def _create_model(self, train_sample):
        past_target = train_sample[0]
        past_covs   = train_sample[1]

        target_dim = past_target.shape[-1]
        past_cov_dim = past_covs.shape[-1] if past_covs is not None else 0
        input_size = target_dim + past_cov_dim
    
        future_target = train_sample[-1]  
        output_size = future_target.shape[-1]
    
        return EnformerModule(
            input_size=input_size,
            output_size=output_size,
            nr_params=getattr(self, "nr_params", 1),
            num_samples_engression=self.num_samples_engression,
            noise_std=self.noise_std,
            noise_dist=self.noise_dist,
            **self.model_params
        )




class GEnformerModule(_TransformerModule):
    def __init__(
        self,
        input_size,
        output_size,
        nr_params,
        edges,
        num_nodes,
        gcn_out_feat,
        node_feat_dim: int,
        num_samples_engression=10,
        noise_dist="gaussian",
        noise_std=1,
        target_coverage=0.95,
        graph_conv_params: typing.Optional[dict] = None,
        lambda_calib=0,
        *args,
        **kwargs
    ):
        self.activation_name = kwargs.get("activation", "relu")
        kwargs.pop("activation", None)
        self.temperature = kwargs.pop("temperature", 10.0)
        self.width_penalty = kwargs.pop("width_penalty", 0.5)
        kwargs.pop("pl_trainer_kwargs", None)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            nr_params=nr_params,
            activation=self.activation_name,
            *args,
            **kwargs
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.M = num_samples_engression
        self.noise_std = noise_std
        self.alpha = target_coverage  # target coverage
        self.lambda_calib = lambda_calib       # calibration strength
        self.noise_dist = noise_dist
        self.edges = edges
        self.num_nodes = num_nodes
        self.gcn_out_feat = gcn_out_feat

        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "add",
                "activation": None,
            }

        # graph convolution
        gcn_activation = graph_conv_params.get("activation", None)
        self.gcn = GraphConv(
            in_feat=node_feat_dim,
            out_feat=gcn_out_feat,
            edges=self.edges,
            num_nodes=self.num_nodes,
            aggregation_type=graph_conv_params.get("aggregation_type", "mean"),
            combination_type=graph_conv_params.get("combination_type", "add"),
            activation=gcn_activation
        ).to(device)

        # noise injection
        if self.noise_dist == "gaussian":
            self.encoder = nn.Sequential(GaussianNoise(self.noise_std, 42).to(device), self.encoder)
        elif self.noise_dist == "uniform":
            self.encoder = nn.Sequential(UniformNoise(self.noise_std, 42).to(device), self.encoder)
        else:
            raise ValueError("noise_dist must be either `gaussian` or `uniform`.")

    def forward(self, x_in, *args, **kwargs):
        x_in_float = tuple(
            t.float() if isinstance(t, torch.Tensor) and torch.is_floating_point(t) else t 
            for t in x_in
        )
        return super().forward(x_in_float, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        past_target = batch[0].to(device)       # (B, T, F)
        past_covs = batch[1].to(device) if batch[1] is not None else None
        static_covs = batch[4].to(device) if batch[4] is not None else None
        future_target = batch[-1].to(device)    # (B, T_out, F)

        # recover (N, D) structure
        B, T, F = past_target.shape
        N = self.num_nodes
        D = F // N

        past_target = past_target.view(B, T, N, D)
        T_out = future_target.shape[1]
        future_target = future_target.view(B, T_out, N, D)

        # apply GCN - (N, B, T, D)
        x = past_target.permute(2, 0, 1, 3)

        x = self.gcn(x)  # (N, B, T, gcn_out)

        # back to (B, T, N, gcn_out)
        x = x.permute(1, 2, 0, 3)

        # flatten for Transformer
        B, T, N, Dg = x.shape
        x = x.reshape(B, T, N * Dg)

        # engression sampling
        x_m = x.repeat_interleave(self.M, dim=0)

        if past_covs is not None:
            past_covs_m = past_covs.repeat_interleave(self.M, dim=0).to(device)
            data_m = torch.cat([x_m, past_covs_m], dim=2)
        else:
            data_m = x_m

        fut_covs_m = batch[3].to(device).repeat_interleave(self.M, dim=0) if batch[3] is not None else None
        static_covs_m = static_covs.repeat_interleave(self.M, dim=0) if static_covs is not None else None

        # forward
        x_expanded = (data_m.float(), fut_covs_m, static_covs_m)
        y_hat_raw = self(x_expanded)  # (B*M, T_out, F_out)

        # reshape output back
        y_hat_raw = y_hat_raw.view(B, self.M, T_out, N, self.gcn_out_feat)
        samples = y_hat_raw.permute(1, 0, 2, 3, 4)  # (M, B, T_out, N, D)

        # energy score
        es_loss = energy_score_loss_st(future_target, samples)

        # calibration (only if enabled)
        if self.lambda_calib > 0:
            cal_loss, coverage = calibration_loss(
                future_target,
                samples,
                alpha=self.alpha,
                temperature=self.temperature,
                width_penalty=self.width_penalty
            )
            loss = es_loss + self.lambda_calib * cal_loss
        else:
            loss = es_loss
            cal_loss = torch.tensor(0.0, device=es_loss.device)
            coverage = torch.tensor(float("nan"), device=es_loss.device)

        # final loss
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(1e6, device=future_target.device)

        # logging
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_es", es_loss, on_epoch=True)
        self.log("train_calib", cal_loss, on_epoch=True)
        self.log("train_coverage", coverage, prog_bar=True, on_epoch=True)
        self.log("noise_std", self.noise_std, on_epoch=True)

        return loss


class GEnformer(TransformerModel):
    r"""
    Graph-Enformer (GEnformer): Spatiotemporal Probabilistic Forecasting.
    
    An extension of the Enformer architecture for spatiotemporal contexts where 
    spatial locations are represented as nodes in an interconnected graph. This model 
    jointly captures temporal dynamics, complex spatial interactions, and predictive 
    uncertainty.

    Before noise injection and Transformer processing, GEnformer applies a 
    Graph Convolutional Network (GCN) layer to map the target observations and the 
    spatial topology (defined by an adjacency matrix) to spatially-aware latent embeddings.
    It then optimizes the Energy Score alongside an optional calibration objective.

    Args:
        input_chunk_length (int): The length of the historical look-back window (\( p \)).
        output_chunk_length (int): The prediction horizon (\( q \)).
        edges (torch.Tensor or List): Graph edges defining the connectivity between nodes.
        num_nodes (int): The number of spatial locations (\( D \)).
        output_chunk_shift (int, optional): Shift for the output chunk. Defaults to 0.
        gcn_out_feat (int, optional): The dimensionality of the latent spatial 
            embeddings outputted by the GCN. Defaults to 32.
        graph_conv_params (dict, optional): Dictionary of parameters for the `GraphConv` layer
            (e.g., aggregation and combination types). Defaults to None.
        num_samples_engression (int, optional): The number of in-sample forecast 
            trajectories (\( M \)) for the ensemble. Defaults to 10.
        noise_dist (str, optional): The type of noise to inject ('gaussian' or 'uniform').
            Defaults to 'gaussian'.
        target_coverage (float, optional): The target prediction interval coverage 
            used for the calibration loss term. Defaults to 0.9.
        noise_std (float, optional): The standard deviation/scale (\( \sigma \)) of 
            the injected noise. Defaults to 1.
        random_state (int, optional): Seed for reproducibility. Defaults to 23.
        save_checkpoints (bool, optional): Whether to save PyTorch Lightning checkpoints.
            Defaults to False.
        lambda_calib (float, optional): The weight of the calibration loss term in 
            the overall optimization objective. Defaults to 2.
        **kwargs: Additional parameters passed to the underlying `TransformerModel`.
    """
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        edges,
        num_nodes: int,
        output_chunk_shift: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: typing.Union[str, torch.nn.Module, None] = None,
        custom_encoder: typing.Optional[torch.nn.Module] = None,
        custom_decoder: typing.Optional[torch.nn.Module] = None,
        gcn_out_feat: int = 32,
        graph_conv_params: typing.Optional[dict] = None,
        num_samples_engression: int = 10,
        noise_dist: str = "gaussian",
        target_coverage: float = 0.9,
        noise_std: float = 1,
        random_state: int = 23,
        save_checkpoints: bool = False,
        lambda_calib: float = 2,
        **kwargs
    ):
        self.num_samples_engression = num_samples_engression
        self.noise_std = noise_std
        self.noise_dist = noise_dist
        self.edges = edges
        self.target_coverage = target_coverage
        self.num_nodes = num_nodes
        self.gcn_out_feat = gcn_out_feat
        self.random_state = random_state
        self.save_checkpoints = save_checkpoints
        self.graph_conv_params = graph_conv_params
        self.lambda_calib = lambda_calib
        self.temperature = kwargs.get("temperature", 5.0)
        self.width_penalty = kwargs.get("width_penalty", 0.01)

        for k in ["temperature", "width_penalty", "target_coverage", "lambda_calib"]:
            kwargs.pop(k, None)

        # Lightning Trainer arguments
        pl_kwargs = {
            "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "deterministic": False,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": True
        }

        pl_trainer_kwargs = kwargs.pop("pl_trainer_kwargs", None)
        if pl_trainer_kwargs is not None:
            pl_kwargs.update(pl_trainer_kwargs)

        # Determine whether to allow lightning to handle epoch checkpoints
        if save_checkpoints:
            pl_kwargs["enable_checkpointing"] = True
            pl_kwargs["default_root_dir"] = "checkpoints"
        else:
            pl_kwargs["enable_checkpointing"] = False

        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["temperature", "width_penalty", "target_coverage"]}

        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            random_state=random_state,
            pl_trainer_kwargs=pl_kwargs,
            **safe_kwargs
        )

        keys_to_remove = [
            'edges', 'num_nodes', 'gcn_out_feat', 'num_samples_engression',
            'noise_dist', 'noise_std', 'graph_conv_params', 'batch_size',
            'n_epochs', 'random_state', 'save_checkpoints', 'node_feat_dim',
            'target_coverage', 'temperature', 'width_penalty'
        ]
        for key in keys_to_remove:
            self.model_params.pop(key, None)

    def _create_model(self, train_sample):
        past_target = train_sample[0]
        target_dim = past_target.shape[-1]   # F = N * D
        N = self.num_nodes
        assert target_dim % N == 0, f"Target dimension {target_dim} must be divisible by num_nodes {N}"
        D = target_dim // N
        past_covs = train_sample[1]
        cov_dim = past_covs.shape[-1] if past_covs is not None else 0
        input_size = self.gcn_out_feat * self.num_nodes + cov_dim

        future_target = train_sample[-1]
        output_size = self.num_nodes * self.gcn_out_feat

        # Create a clean copy of model_params for the Module
        module_params = self.model_params.copy()
        module_params.pop("pl_trainer_kwargs", None)

        return GEnformerModule(
            input_size=input_size,
            output_size=output_size,
            nr_params=getattr(self, "nr_params", 1),
            edges=self.edges,
            num_nodes=self.num_nodes,
            gcn_out_feat=self.gcn_out_feat,
            node_feat_dim=D,
            num_samples_engression=self.num_samples_engression,
            noise_std=self.noise_std,
            noise_dist=self.noise_dist,
            graph_conv_params=self.graph_conv_params,
            target_coverage=self.target_coverage,
            temperature=self.temperature,
            width_penalty=self.width_penalty,
            **module_params
        )


