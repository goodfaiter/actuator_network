"""Tests for the frozen spring transformer deployment script."""

import os
import tempfile

import numpy as np
import pytest
import torch
from mcap_ros2.reader import read_ros2_messages

from actuator_network.test_spring_transformer_frozen import run_frozen_transformer_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _build_frozen_checkpoint(tmpdir: str) -> tuple[str, torch.nn.Module, torch.Tensor]:
    """Build a tiny pruned frozen-latent deployment checkpoint in-memory.

    Returns:
        Tuple of (checkpoint path, spring coefficient head module, embedded latent).
    """
    from actuator_network.export_frozen_latent import build_frozen_latent_model
    from actuator_network.helpers.torch_model import (
        SpringCoefficientHead,
        SpringTransformerModel,
        TorchTransformerModel,
    )
    from actuator_network.helpers.wrapper import ScaledModelWrapper

    device = torch.device("cpu")
    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=4,
        num_layers=1,
        history_size=4,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + 4,
        output_size=1,
        num_layers=1,
        history_size=1,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=4, device=device)
    estimator = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=4,
    )
    wrapped = ScaledModelWrapper(
        estimator,
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        frequency=200,
        history_size=1,
        stride=20,
        input_columns=["measured_position_rad_data", "measured_velocity_rad_per_sec_data"],
        output_columns=["tendon_bota_force_newton_data", "spring_coeff"],
    )
    wrapped.eval()
    full_path = os.path.join(tmpdir, "spring_full.pt")
    torch.jit.script(wrapped).save(full_path)

    loaded_full = torch.jit.load(full_path)
    frozen_latent = torch.randn(1, 1, 4)
    pruned = build_frozen_latent_model(loaded_full, frozen_latent)
    wrapped_pruned = ScaledModelWrapper(
        pruned,
        loaded_full.input_mean,
        loaded_full.input_std,
        loaded_full.output_mean,
        loaded_full.output_std,
        frequency=loaded_full.metadata["frequency"],
        history_size=loaded_full.metadata["history_size"],
        stride=loaded_full.metadata["stride"],
        input_columns=loaded_full.input_columns,
        output_columns=loaded_full.output_columns,
    )
    wrapped_pruned.eval()
    model_path = os.path.join(tmpdir, "spring_pruned.pt")
    torch.jit.script(wrapped_pruned).save(model_path)
    return model_path, spring_coeff_head, frozen_latent


def test_run_frozen_transformer_inference_writes_predictions(tmp_path):
    """The frozen deployment script pins the spring channel to the embedded latent."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, spring_coeff_head, frozen_latent = _build_frozen_checkpoint(tmpdir)

        output_paths = run_frozen_transformer_inference(model_path, [TEST_MCAP])

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_spring_transformer_frozen_predicted.mcap")

        # The embedded latent is constant: the spring channel must be constant
        # and equal the denormalized coefficient head output of that latent.
        expected_spring = float(spring_coeff_head(frozen_latent)[0, 0, 0])
        spring_values = [
            message.ros_msg.data for message in read_ros2_messages(output_paths[0], topics=["/spring_coeff_predicted"])
        ]
        assert len(spring_values) > 0
        assert np.allclose(spring_values, expected_spring, atol=1e-5)

        # The force channel varies with the live dynamics but is populated.
        force_values = [
            message.ros_msg.data
            for message in read_ros2_messages(output_paths[0], topics=["/tendon_bota_force_newton_data_predicted"])
        ]
        assert not np.allclose(force_values, 0.0)


def test_run_frozen_transformer_inference_rejects_full_checkpoint(tmp_path):
    """The dedicated frozen script requires a pruned deployment checkpoint."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        from actuator_network.helpers.torch_model import (
            SpringCoefficientHead,
            SpringTransformerModel,
            TorchTransformerModel,
        )
        from actuator_network.helpers.wrapper import ScaledModelWrapper

        device = torch.device("cpu")
        model_transformer = TorchTransformerModel(
            input_size=2,
            output_size=4,
            num_layers=1,
            history_size=4,
            num_heads=2,
            hidden_dim=8,
            device=device,
        )
        force_transformer = TorchTransformerModel(
            input_size=2 + 4,
            output_size=1,
            num_layers=1,
            history_size=1,
            num_heads=2,
            hidden_dim=8,
            device=device,
        )
        spring_coeff_head = SpringCoefficientHead(latent_dim=4, device=device)
        estimator = SpringTransformerModel(
            model_transformer=model_transformer,
            force_transformer=force_transformer,
            spring_coeff_head=spring_coeff_head,
            latent_dim=4,
        )
        wrapped = ScaledModelWrapper(
            estimator,
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            frequency=200,
            history_size=1,
            stride=20,
            input_columns=["measured_position_rad_data", "measured_velocity_rad_per_sec_data"],
            output_columns=["tendon_bota_force_newton_data", "spring_coeff"],
        )
        wrapped.eval()
        full_model_path = os.path.join(tmpdir, "spring_full.pt")
        torch.jit.script(wrapped).save(full_model_path)

        with pytest.raises(ValueError) as exc_info:
            run_frozen_transformer_inference(full_model_path, [TEST_MCAP])
        assert "pruned" in str(exc_info.value)
