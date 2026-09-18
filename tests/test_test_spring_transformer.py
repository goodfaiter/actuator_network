"""Tests for the spring transformer inference script."""

import os
import tempfile

import numpy as np
import pytest
import torch
from mcap_ros2.reader import read_ros2_messages

from actuator_network.test_spring_transformer import run_spring_transformer_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def test_run_spring_transformer_inference_writes_predictions(tmp_path):
    """Inference at the model's inference rate writes a populated predicted column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # A tiny scripted spring-model wrapper is built in-memory so the run
        # function is exercised end to end without a training checkpoint.
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
        model_path = os.path.join(tmpdir, "spring.pt")
        torch.jit.script(wrapped).save(model_path)

        output_paths = run_spring_transformer_inference(model_path, [TEST_MCAP])

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_spring_transformer_predicted.mcap")

        # The predicted recording holds only the inferred samples; per topic the
        # timestamps are spaced at the model's inference rate (200 Hz / 20),
        # expressed in nanoseconds.
        timestamps = [
            message.log_time_ns
            for message in read_ros2_messages(output_paths[0], topics=["/tendon_bota_force_newton_data"])
        ]
        assert len(timestamps) > 1

        spacings = np.diff(timestamps)
        assert np.median(spacings) == pytest.approx(1e9 / (200 / 20), rel=0.05)
