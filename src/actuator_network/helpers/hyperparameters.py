"""Hyperparameter configuration and sweep helpers for training scripts.

This module stores the default training configuration and helpers for building
it from a W&B sweep configuration.
"""

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class EstimatedSpringTransformerConfig:
    """Hyperparameters for the estimated-spring transformer training pipeline.

    The defaults match the current hardcoded values in
    ``train_estimated_spring_transformer.py``.
    """

    # Data/build parameters
    data_freq: int = 200
    prediction: bool = False
    velocity_threshold: float = 0.5

    # Spring transformer parameters
    spring_history_size: int = 600
    spring_stride: int = 4
    spring_num_layers: int = 2
    spring_num_heads: int = 2
    spring_hidden_dim: int = 32
    spring_dropout: float = 0.2
    spring_activation: str = "relu"

    # Force transformer parameters
    force_history_size: int = 150
    force_stride: int = 2
    force_num_layers: int = 2
    force_num_heads: int = 2
    force_hidden_dim: int = 32
    force_dropout: float = 0.1
    force_activation: str = "relu"

    # Training parameters
    num_epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 512
    accumulation_steps: int = 2
    aux_weight: float = 1.0
    weight_decay: float = 1e-5
    scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    input_noise_std: float = 0.01
    spring_alpha: float = 0.05
    val_fraction: float = 0.2

    @classmethod
    def defaults(cls) -> "EstimatedSpringTransformerConfig":
        """Return the default configuration."""
        return cls()

    def is_valid(self) -> bool:
        """Return True if the configuration is valid for the Transformers.

        Both spring and force transformers require ``hidden_dim`` to be
        divisible by ``num_heads``.
        """
        spring_valid = self.spring_hidden_dim % self.spring_num_heads == 0
        force_valid = self.force_hidden_dim % self.force_num_heads == 0
        return spring_valid and force_valid

    @classmethod
    def from_wandb_config(cls, cfg: Any) -> "EstimatedSpringTransformerConfig":
        """Build a configuration from a W&B sweep config dict-like object.

        Values present in ``cfg`` override the defaults. The sweep may expose
        reparameterized parameters that are converted back to the dataclass
        fields:

        - ``*_hidden_dim_per_head`` -> ``*_hidden_dim = num_heads * per_head``
        - ``spring_stride_multiplier`` -> ``spring_stride = force_stride * multiplier``

        Explicit fields take precedence if both the direct and reparameterized
        versions are provided.

        Args:
            cfg: A dict-like object (e.g., ``wandb.config``) providing sweep
                hyperparameters.

        Returns:
            An ``EstimatedSpringTransformerConfig`` instance.
        """
        defaults = cls.defaults()
        kwargs: dict[str, Any] = {}

        for f in fields(cls):
            if f.name in cfg:
                kwargs[f.name] = cfg[f.name]
            else:
                kwargs[f.name] = getattr(defaults, f.name)

        # Compute hidden_dim from per_head * num_heads when the sweep uses the
        # reparameterized parameters. Explicit hidden_dim takes precedence if
        # both are provided.
        if "spring_hidden_dim" not in cfg and "spring_hidden_dim_per_head" in cfg:
            spring_heads = cfg.get("spring_num_heads", kwargs["spring_num_heads"])
            kwargs["spring_hidden_dim"] = spring_heads * cfg["spring_hidden_dim_per_head"]

        if "force_hidden_dim" not in cfg and "force_hidden_dim_per_head" in cfg:
            force_heads = cfg.get("force_num_heads", kwargs["force_num_heads"])
            kwargs["force_hidden_dim"] = force_heads * cfg["force_hidden_dim_per_head"]

        # Compute spring_stride from force_stride * multiplier so that
        # spring_stride is always a multiple of force_stride.
        if "spring_stride" not in cfg and "spring_stride_multiplier" in cfg:
            force_stride = cfg.get("force_stride", kwargs["force_stride"])
            kwargs["spring_stride"] = force_stride * cfg["spring_stride_multiplier"]

        return cls(**kwargs)
