import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    """
    Gaussian noise injection module for the engression paradigm.
    
    This module injects additive Gaussian noise into the continuous representations
    of the sequence, acting as a pre-additive stochastic noise layer.
    According to the paper "Deep Generative Transformers for Probabilistic Time Series 
    and Spatiotemporal Forecasting", injecting this stochastic noise enables the 
    Transformer architecture to learn the full conditional predictive distribution 
    rather than point predictions.
    
    Args:
        std (float): The standard deviation (\\(\sigma\\)) of the Gaussian noise.
        seed (int | None): An optional seed for reproducible noise generation.
        
    Mathematical Definition:
        The noise \\(\epsilon^{(m)}\\) is sampled from a Gaussian distribution:
        \\( \epsilon^{(m)} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}) \\)
        The noise is then added to the input sequence:
        \\( \mathbf{X'}_{batch}^{(m)} = \mathbf{X}_{batch}^{(m)} + \epsilon^{(m)} \\)
    """
    def __init__(self, std: float, seed: int | None = None):
        super().__init__()
        self.std = std
        self.seed = seed
        self._gen = None
        self._gen_device = None

    def _get_generator(self, device: torch.device):
        if self._gen is None or self._gen_device != device:
            self._gen = torch.Generator(device=device)
            self._gen_device = device
            if self.seed is not None:
                self._gen.manual_seed(self.seed)
        return self._gen

    def reset_seed(self, seed: int | None = None):
        """Reset the internal generator seed."""
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            raise ValueError("No seed set for this module.")
        if self._gen is not None:
            self._gen.manual_seed(self.seed)

    def reset_std(self, std: float | None = None):
        """Update the standard deviation of the noise."""
        self.std = std

    def forward(self, x: torch.Tensor):
        """
        Injects Gaussian noise into the input tensor.
        
        Args:
            x (torch.Tensor): The input sequence representations.
            
        Returns:
            torch.Tensor: The noise-perturbed representations.
        """
        g = self._get_generator(x.device)
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=g,
        )
        return x + noise * self.std


class UniformNoise(nn.Module):
    """
    Uniform noise injection module for the engression paradigm.
    
    This module injects additive Uniform noise into the continuous representations
    of the sequence.
    
    Args:
        std (float): The scale parameter (\\(\sigma\\)) defining the noise boundaries.
        seed (int | None): An optional seed for reproducible noise generation.
        
    Mathematical Definition:
        The noise \\(\epsilon^{(m)}\\) is sampled from a Uniform distribution:
        \\( \epsilon^{(m)} \sim \mathcal{U}(-\sigma, \sigma) \\)
        The noise is then added to the input sequence:
        \\( \mathbf{X'}_{batch}^{(m)} = \mathbf{X}_{batch}^{(m)} + \epsilon^{(m)} \\)
    """
    def __init__(self, std: float, seed: int | None = None):
        super().__init__()
        self.std = std
        self.seed = seed
        self._gen = None
        self._gen_device = None

    def _get_generator(self, device: torch.device):
        if self._gen is None or self._gen_device != device:
            self._gen = torch.Generator(device=device)
            self._gen_device = device
            if self.seed is not None:
                self._gen.manual_seed(self.seed)
        return self._gen

    def reset_seed(self, seed: int | None = None):
        """Reset the internal generator seed."""
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            raise ValueError("No seed set for this module.")
        if self._gen is not None:
            self._gen.manual_seed(self.seed)

    def reset_std(self, std: float | None = None):
        """Update the standard deviation parameter."""
        self.std = std

    def forward(self, x: torch.Tensor):
        """
        Injects Uniform noise into the input tensor.
        
        Args:
            x (torch.Tensor): The input sequence representations.
            
        Returns:
            torch.Tensor: The noise-perturbed representations.
        """
        g = self._get_generator(x.device)
        noise = torch.rand(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=g,
        )
        return x + (2 * self.std) * noise - self.std
