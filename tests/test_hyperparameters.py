"""Tests for the hyperparameter configuration helpers."""

from actuator_network.helpers.hyperparameters import EstimatedSpringTransformerConfig


def test_defaults_match_expected_values():
    """The default config should match the current training-script defaults."""
    config = EstimatedSpringTransformerConfig.defaults()
    assert config.num_epochs == 50
    assert config.learning_rate == 0.001
    assert config.batch_size == 512
    assert config.spring_history_size == 600
    assert config.spring_stride == 4
    assert config.force_history_size == 150
    assert config.force_stride == 2
    assert config.spring_num_layers == 2
    assert config.spring_num_heads == 2
    assert config.spring_hidden_dim == 32
    assert config.spring_activation == "relu"
    assert config.force_num_layers == 2
    assert config.force_num_heads == 2
    assert config.force_hidden_dim == 32
    assert config.force_activation == "relu"
    assert config.val_fraction == 0.2


def test_is_valid_requires_divisible_heads():
    """Valid configs require hidden_dim to be divisible by num_heads for both transformers."""
    assert EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=4, force_hidden_dim=64, force_num_heads=8
    ).is_valid()
    assert not EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=6, force_hidden_dim=64, force_num_heads=8
    ).is_valid()
    assert not EstimatedSpringTransformerConfig(
        spring_hidden_dim=32, spring_num_heads=4, force_hidden_dim=64, force_num_heads=6
    ).is_valid()


def test_from_wandb_config_uses_defaults_for_missing_keys():
    """Building from an empty config-like dict should return defaults."""
    config = EstimatedSpringTransformerConfig.from_wandb_config({})
    assert config == EstimatedSpringTransformerConfig.defaults()


def test_from_wandb_config_applies_overrides():
    """Provided values should override defaults."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "learning_rate": 0.123,
            "batch_size": 1024,
            "spring_num_layers": 3,
        }
    )
    assert config.learning_rate == 0.123
    assert config.batch_size == 1024
    assert config.spring_num_layers == 3
    # Unspecified fields remain at their default.
    assert config.spring_history_size == 600


def test_from_wandb_config_reparameterizes_hidden_dim():
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


def test_from_wandb_config_explicit_hidden_dim_wins():
    """An explicit *_hidden_dim should override the per_head computation."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "spring_num_heads": 4,
            "spring_hidden_dim_per_head": 8,
            "spring_hidden_dim": 40,
        }
    )
    assert config.spring_hidden_dim == 40


def test_from_wandb_config_reparameterizes_spring_stride():
    """When spring_stride_multiplier is provided, spring_stride is computed from force_stride."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "force_stride": 3,
            "spring_stride_multiplier": 4,
        }
    )
    assert config.spring_stride == 12
    assert config.spring_stride % config.force_stride == 0


def test_from_wandb_config_explicit_spring_stride_wins():
    """An explicit spring_stride should override the multiplier computation."""
    config = EstimatedSpringTransformerConfig.from_wandb_config(
        {
            "force_stride": 3,
            "spring_stride_multiplier": 4,
            "spring_stride": 5,
        }
    )
    assert config.spring_stride == 5
