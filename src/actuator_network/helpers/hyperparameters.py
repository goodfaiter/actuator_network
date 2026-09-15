"""Hyperparameter configuration and sweep helpers for training scripts.

This module stores the default training configurations and helpers for building
configurations from a W&B sweep configuration.
"""

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class BaseTrainingConfig:
    """Base hyperparameters shared across all training pipelines.

    Derived classes add model-specific and data-specific fields. All fields are
    intended to be overridable from ``wandb.config`` so that every training
    script can be used as the target program for a W&B sweep agent.
    """

    # Data/build parameters
    data_freq: int = 200
    prediction: bool = False

    # Training parameters
    num_epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 1024
    weight_decay: float = 0.0
    scheduler_type: str = "none"
    max_grad_norm: float = 1.0
    val_fraction: float = 1.0

    @classmethod
    def defaults(cls) -> "BaseTrainingConfig":
        """Return the default configuration."""
        return cls()

    def is_valid(self) -> bool:
        """Return True if the configuration is valid.

        Derived classes should override this to enforce model-specific
        constraints.
        """
        return True

    @classmethod
    def from_wandb_config(cls, cfg: Any) -> "BaseTrainingConfig":
        """Build a configuration from a W&B sweep config dict-like object.

        Values present in ``cfg`` override the defaults. Unspecified fields keep
        their default values.

        Args:
            cfg: A dict-like object (e.g., ``wandb.config``) providing sweep
                hyperparameters.

        Returns:
            A config instance of the calling class.
        """
        defaults = cls.defaults()
        kwargs: dict[str, Any] = {}

        for f in fields(cls):
            if f.name in cfg:
                kwargs[f.name] = cfg[f.name]
            else:
                kwargs[f.name] = getattr(defaults, f.name)

        return cls(**kwargs)


@dataclass
class MlpConfig(BaseTrainingConfig):
    """Hyperparameters for the MLP training pipeline.

    The defaults match the current hardcoded values in ``train_mlp.py``.
    """

    # Data/build parameters
    data_freq: int = 80
    num_hist: int = 30
    stride: int = 4

    # Model parameters
    hidden_layers: list[int] = field(default_factory=lambda: [256, 64, 16])

    # Data columns
    input_cols: list[str] = field(
        default_factory=lambda: ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    )
    output_cols: list[str] = field(default_factory=lambda: ["tendon_bota_force_newton_data"])


@dataclass
class RnnConfig(BaseTrainingConfig):
    """Hyperparameters for the RNN training pipeline.

    The defaults match the current hardcoded values in ``train_rnn.py``.
    """

    # Data/build parameters
    data_freq: int = 80
    seq_length: int = 512
    stride: int = 1

    # Model parameters
    hidden_size: int = 64
    num_layers: int = 4
    dropout: float = 0.1

    # Training parameters
    num_epochs: int = 50
    chunk_batch_size: int = 4

    # Data columns
    input_cols: list[str] = field(
        default_factory=lambda: ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    )
    output_cols: list[str] = field(default_factory=lambda: ["tendon_bota_force_newton_data"])


@dataclass
class TransformerConfig(BaseTrainingConfig):
    """Hyperparameters for the Transformer training pipeline.

    The defaults match the current hardcoded values in ``train_transformer.py``.
    """

    # Data/build parameters
    history_size: int = 150
    stride: int = 2

    @property
    def inference_freq(self) -> int:
        """Return the effective inference frequency in Hz."""
        return self.data_freq // self.stride

    # Model parameters
    num_layers: int = 2
    num_heads: int = 4
    hidden_dim: int = 32
    dropout: float = 0.1
    activation: str = "relu"

    # Training parameters
    num_epochs: int = 50

    # Data columns
    input_cols: list[str] = field(
        default_factory=lambda: ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    )
    output_cols: list[str] = field(default_factory=lambda: ["tendon_bota_force_newton_data"])

    def is_valid(self) -> bool:
        """Return True if the Transformer hidden dimension is valid."""
        return self.hidden_dim % self.num_heads == 0 and self.hidden_dim % 2 == 0

    @classmethod
    def from_wandb_config(cls, cfg: Any) -> "TransformerConfig":
        """Build a configuration from a W&B sweep config.

        Supports reparameterization via ``hidden_dim_per_head``:
        ``hidden_dim = num_heads * hidden_dim_per_head``.
        """
        config = super().from_wandb_config(cfg)
        if "hidden_dim" not in cfg and "hidden_dim_per_head" in cfg:
            config.hidden_dim = config.num_heads * cfg["hidden_dim_per_head"]
        return config


@dataclass
class TransformerAutoregressiveConfig(TransformerConfig):
    """Hyperparameters for the autoregressive Transformer training pipeline.

    The defaults match the current hardcoded values in
    ``train_transformer_autoregressive.py``.
    """

    # Data columns include the previous force as an autoregressive channel.
    input_cols: list[str] = field(
        default_factory=lambda: [
            "delta_position_rad_data",
            "measured_velocity_rad_per_sec_data",
            "tendon_bota_force_newton_data",
        ]
    )
    output_cols: list[str] = field(default_factory=lambda: ["tendon_bota_force_newton_data"])


@dataclass
class M5FrictionConfig(BaseTrainingConfig):
    """Hyperparameters for fitting the M5 friction model.

    The defaults match the current hardcoded values in ``train_m5.py``.
    """

    # Data/build parameters
    data_freq: int = 200

    # Training parameters
    num_epochs: int = 2000
    learning_rate: float = 0.01
    patience: int = 200
    trainable_motor_gain: bool = False


@dataclass
class M5TransformerConfig(TransformerConfig):
    """Hyperparameters for the M5 + Transformer physics-coupled pipeline.

    The defaults match the current hardcoded values in
    ``train_m5_transformer.py``.
    """

    # M5-specific parameters
    m5_trainable: bool = False
    motor_gain_trainable: bool = False
    aux_weight: float = 0.0

    # Training parameters
    num_epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 1024
    val_fraction: float = 1.0

    # Output channels produced by the physics-coupled model.
    model_output_cols: list[str] = field(
        default_factory=lambda: [
            "tendon_bota_force_newton_data",
            "tau_motor_newton_data",
            "tau_friction_newton_data",
            "tau_external_pred_newton_data",
        ]
    )


@dataclass
class EstimatedSpringTransformerConfig(BaseTrainingConfig):
    """Hyperparameters for the estimated-spring transformer training pipeline.

    The defaults match the current hardcoded values in
    ``train_estimated_spring_transformer.py``.
    """

    # Data/build parameters
    data_freq: int = 200
    prediction: bool = False
    velocity_threshold: float = 0.5

    # Spring transformer parameters
    spring_history_size: int = 500
    spring_stride: int = 4  # force_stride (2) * spring_stride_multiplier (2)
    spring_num_layers: int = 1
    spring_num_heads: int = 4
    spring_hidden_dim: int = 112  # spring_num_heads (4) * spring_hidden_dim_per_head (28)
    spring_latent_dim: int = 16
    spring_dropout: float = 0.3
    spring_activation: str = "relu"

    # Force transformer parameters
    force_history_size: int = 50
    force_stride: int = 2
    force_num_layers: int = 2
    force_num_heads: int = 4
    force_hidden_dim: int = 72  # force_num_heads (4) * force_hidden_dim_per_head (18)
    force_dropout: float = 0.3
    force_activation: str = "relu"

    # Training parameters
    num_epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 1024
    accumulation_steps: int = 1
    aux_weight: float = 1.0
    weight_decay: float = 1e-5
    scheduler_type: str = "none"
    max_grad_norm: float = 1.0
    input_noise_std: float = 0.05
    spring_alpha: float = 0.1
    val_fraction: float = 1.0

    def is_valid(self) -> bool:
        """Return True if the configuration is valid for the Transformers.

        Both spring and force transformers require ``hidden_dim`` to be
        divisible by ``num_heads`` and even (the positional encoding assumes
        an even number of dimensions).
        """
        spring_valid = self.spring_hidden_dim % self.spring_num_heads == 0 and self.spring_hidden_dim % 2 == 0
        force_valid = self.force_hidden_dim % self.force_num_heads == 0 and self.force_hidden_dim % 2 == 0
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
        config = super().from_wandb_config(cfg)

        # Compute hidden_dim from per_head * num_heads when the sweep uses the
        # reparameterized parameters. Explicit hidden_dim takes precedence if
        # both are provided.
        if "spring_hidden_dim" not in cfg and "spring_hidden_dim_per_head" in cfg:
            spring_heads = cfg.get("spring_num_heads", config.spring_num_heads)
            config.spring_hidden_dim = spring_heads * cfg["spring_hidden_dim_per_head"]

        if "force_hidden_dim" not in cfg and "force_hidden_dim_per_head" in cfg:
            force_heads = cfg.get("force_num_heads", config.force_num_heads)
            config.force_hidden_dim = force_heads * cfg["force_hidden_dim_per_head"]

        # Compute spring_stride from force_stride * multiplier so that
        # spring_stride is always a multiple of force_stride.
        if "spring_stride" not in cfg and "spring_stride_multiplier" in cfg:
            force_stride = cfg.get("force_stride", config.force_stride)
            config.spring_stride = force_stride * cfg["spring_stride_multiplier"]

        return config
