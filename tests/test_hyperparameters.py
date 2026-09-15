"""Tests for the hyperparameter configuration helpers."""

from actuator_network.helpers.hyperparameters import (
    BaseTrainingConfig,
    EstimatedSpringTransformerConfig,
    M5FrictionConfig,
    M5TransformerConfig,
    MlpConfig,
    RnnConfig,
    TransformerAutoregressiveConfig,
    TransformerConfig,
)


def test_base_training_config_defaults():
    """The base config should provide sensible shared defaults."""
    config = BaseTrainingConfig.defaults()
    assert config.data_freq == 200
    assert config.prediction is False
    assert config.num_epochs == 50
    assert config.learning_rate == 0.001
    assert config.batch_size == 1024
    assert config.weight_decay == 0.0
    assert config.scheduler_type == "none"
    assert config.max_grad_norm == 1.0
    assert config.val_fraction == 1.0


def test_base_training_config_is_valid_by_default():
    """The base config should be valid by default."""
    assert BaseTrainingConfig.defaults().is_valid()


def test_mlp_config_defaults():
    """The MLP config defaults should match train_mlp.py."""
    config = MlpConfig.defaults()
    assert config.data_freq == 80
    assert config.num_hist == 30
    assert config.stride == 4
    assert config.hidden_layers == [256, 64, 16]
    assert config.input_cols == ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    assert config.output_cols == ["tendon_bota_force_newton_data"]


def test_rnn_config_defaults():
    """The RNN config defaults should match train_rnn.py."""
    config = RnnConfig.defaults()
    assert config.data_freq == 80
    assert config.seq_length == 512
    assert config.stride == 1
    assert config.hidden_size == 64
    assert config.num_layers == 4
    assert config.dropout == 0.1
    assert config.num_epochs == 50
    assert config.chunk_batch_size == 4


def test_transformer_config_defaults():
    """The Transformer config defaults should match train_transformer.py."""
    config = TransformerConfig.defaults()
    assert config.data_freq == 200
    assert config.history_size == 150
    assert config.stride == 2
    assert config.inference_freq == 100
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.hidden_dim == 32
    assert config.dropout == 0.1
    assert config.activation == "relu"


def test_transformer_config_is_valid_requires_divisible_heads_and_even_dim():
    """Valid Transformer configs require hidden_dim divisible by num_heads and even."""
    assert TransformerConfig(hidden_dim=32, num_heads=4).is_valid()
    assert not TransformerConfig(hidden_dim=32, num_heads=6).is_valid()
    assert not TransformerConfig(hidden_dim=33, num_heads=1).is_valid()


def test_transformer_config_reparameterizes_hidden_dim():
    """When hidden_dim_per_head is provided, hidden_dim is computed from num_heads."""
    config = TransformerConfig.from_wandb_config({"num_heads": 4, "hidden_dim_per_head": 8})
    assert config.hidden_dim == 32
    assert config.is_valid()


def test_transformer_config_explicit_hidden_dim_wins():
    """An explicit hidden_dim should override the per_head computation."""
    config = TransformerConfig.from_wandb_config({"num_heads": 4, "hidden_dim_per_head": 8, "hidden_dim": 40})
    assert config.hidden_dim == 40


def test_transformer_autoregressive_config_inherits_transformer():
    """The autoregressive config should inherit Transformer defaults and add the force input."""
    config = TransformerAutoregressiveConfig.defaults()
    assert config.history_size == 150
    assert config.input_cols == [
        "delta_position_rad_data",
        "measured_velocity_rad_per_sec_data",
        "tendon_bota_force_newton_data",
    ]
    assert config.output_cols == ["tendon_bota_force_newton_data"]


def test_m5_friction_config_defaults():
    """The M5 friction config defaults should match train_m5.py."""
    config = M5FrictionConfig.defaults()
    assert config.data_freq == 200
    assert config.num_epochs == 2000
    assert config.learning_rate == 0.01
    assert config.patience == 200
    assert config.trainable_motor_gain is False


def test_m5_transformer_config_defaults():
    """The M5 + Transformer config defaults should match train_m5_transformer.py."""
    config = M5TransformerConfig.defaults()
    assert config.history_size == 150
    assert config.stride == 2
    assert config.inference_freq == 100
    assert config.m5_trainable is False
    assert config.motor_gain_trainable is False
    assert config.aux_weight == 0.0
    assert config.num_epochs == 50
    assert config.batch_size == 1024
    assert config.val_fraction == 1.0
    assert config.model_output_cols == [
        "tendon_bota_force_newton_data",
        "tau_motor_newton_data",
        "tau_friction_newton_data",
        "tau_external_pred_newton_data",
    ]


def test_estimated_spring_transformer_config_defaults():
    """The estimated-spring config defaults should match train_estimated_spring_transformer.py."""
    config = EstimatedSpringTransformerConfig.defaults()
    assert config.data_freq == 200
    assert config.num_epochs == 20
    assert config.learning_rate == 0.001
    assert config.batch_size == 1024
    assert config.spring_history_size == 500
    assert config.spring_stride == 4
    assert config.force_history_size == 50
    assert config.force_stride == 2
    assert config.spring_num_layers == 1
    assert config.spring_num_heads == 4
    assert config.spring_hidden_dim == 112
    assert config.spring_activation == "relu"
    assert config.force_num_layers == 2
    assert config.force_num_heads == 4
    assert config.force_hidden_dim == 72
    assert config.force_activation == "relu"
    assert config.val_fraction == 1.0


def test_estimated_spring_transformer_config_is_valid_requires_divisible_heads_and_even_dim():
    """Valid configs require hidden_dim to be divisible by num_heads and even."""
    assert EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=4, force_hidden_dim=64, force_num_heads=8
    ).is_valid()
    assert not EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=6, force_hidden_dim=64, force_num_heads=8
    ).is_valid()
    assert not EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=4, force_hidden_dim=64, force_num_heads=6
    ).is_valid()
    # Odd hidden_dim values are rejected even when divisible by num_heads.
    assert not EstimatedSpringTransformerConfig(
        spring_hidden_dim=33, spring_num_heads=1, force_hidden_dim=64, force_num_heads=8
    ).is_valid()
    assert not EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=4, force_hidden_dim=21, force_num_heads=1
    ).is_valid()


def test_from_wandb_config_uses_defaults_for_missing_keys():
    """Building from an empty config-like dict should return defaults for all classes."""
    assert BaseTrainingConfig.from_wandb_config({}) == BaseTrainingConfig.defaults()
    assert MlpConfig.from_wandb_config({}) == MlpConfig.defaults()
    assert RnnConfig.from_wandb_config({}) == RnnConfig.defaults()
    assert TransformerConfig.from_wandb_config({}) == TransformerConfig.defaults()
    assert TransformerAutoregressiveConfig.from_wandb_config({}) == TransformerAutoregressiveConfig.defaults()
    assert M5FrictionConfig.from_wandb_config({}) == M5FrictionConfig.defaults()
    assert M5TransformerConfig.from_wandb_config({}) == M5TransformerConfig.defaults()
    assert EstimatedSpringTransformerConfig.from_wandb_config({}) == EstimatedSpringTransformerConfig.defaults()


def test_from_wandb_config_applies_overrides():
    """Provided values should override defaults."""
    config = TransformerConfig.from_wandb_config(
        {
            "learning_rate": 0.123,
            "batch_size": 512,
            "num_layers": 3,
        }
    )
    assert config.learning_rate == 0.123
    assert config.batch_size == 512
    assert config.num_layers == 3
    # Unspecified fields remain at their default.
    assert config.history_size == 150


def test_estimated_spring_from_wandb_config_reparameterizes_hidden_dim():
    """When *_hidden_dim_per_head is provided, hidden_dim is computed from num_heads."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "spring_num_heads": 4,
            "spring_hidden_dim_per_head": 8,
            "force_num_heads": 8,
            "force_hidden_dim_per_head": 16,
        }
    )
    assert config.spring_hidden_dim == 32
    assert config.force_hidden_dim == 128
    assert config.is_valid()


def test_estimated_spring_from_wandb_config_explicit_hidden_dim_wins():
    """An explicit *_hidden_dim should override the per_head computation."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "spring_num_heads": 4,
            "spring_hidden_dim_per_head": 8,
            "spring_hidden_dim": 40,
        }
    )
    assert config.spring_hidden_dim == 40


def test_estimated_spring_from_wandb_config_reparameterizes_spring_stride():
    """When spring_stride_multiplier is provided, spring_stride is computed from force_stride."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "force_stride": 3,
            "spring_stride_multiplier": 4,
        }
    )
    assert config.spring_stride == 12
    assert config.spring_stride % config.force_stride == 0


def test_estimated_spring_from_wandb_config_explicit_spring_stride_wins():
    """An explicit spring_stride should override the multiplier computation."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "force_stride": 3,
            "spring_stride_multiplier": 4,
            "spring_stride": 5,
        }
    )
    assert config.spring_stride == 5
