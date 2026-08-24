import os
import tempfile

import torch

from actuator_network.helpers.torch_model import TorchTransformerModel
from actuator_network.helpers.wrapper import ScaledModelWrapper
from actuator_network.test_transformer_autoregressive import run_transformer_autoregressive_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _make_dummy_scripted_model(tmpdir: str) -> str:
    """Build and save a tiny scripted autoregressive Transformer wrapper."""
    device = torch.device("cpu")

    history_size = 1
    transformer = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )

    input_mean = torch.zeros(1, 3)
    input_std = torch.ones(1, 3)
    output_mean = torch.zeros(1, 1)
    output_std = torch.ones(1, 1)

    wrapped = ScaledModelWrapper(
        transformer,
        input_mean,
        input_std,
        output_mean,
        output_std,
        frequency=80,
        history_size=history_size,
        stride=1,
        prediction=False,
        input_columns=[
            "delta_position_rad_data",
            "measured_velocity_rad_per_sec_data",
            "tendon_bota_force_newton_data",
        ],
        output_columns=["tendon_bota_force_newton_data"],
    )
    wrapped.eval()

    scripted = torch.jit.script(wrapped)
    model_path = os.path.join(tmpdir, "transformer_autoregressive.pt")
    scripted.save(model_path)
    return model_path


def test_run_transformer_autoregressive_inference_creates_output():
    """Inference should create an MCAP with a populated _predicted column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _make_dummy_scripted_model(tmpdir)
        output_paths = run_transformer_autoregressive_inference(model_path, [TEST_MCAP], data_freq=80)

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_transformer_autoregressive_predicted.mcap")
