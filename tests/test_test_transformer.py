import os
import tempfile

import torch

from actuator_network.helpers.torch_model import TorchTransformerModel
from actuator_network.helpers.wrapper import ScaledModelWrapper
from actuator_network.test_transformer import run_transformer_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _make_dummy_scripted_model(tmpdir: str) -> str:
    """Build and save a tiny scripted Transformer wrapper for inference tests."""
    device = torch.device("cpu")

    # Small history size so the test MCAP is guaranteed to produce predictions.
    history_size = 1
    transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )

    input_mean = torch.zeros(1, 2)
    input_std = torch.ones(1, 2)
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
        input_columns=["delta_position_rad_data", "measured_velocity_rad_per_sec_data"],
        output_columns=["tendon_bota_force_newton_data"],
    )
    wrapped.eval()
    transformer.eval()

    scripted = torch.jit.script(wrapped)
    model_path = os.path.join(tmpdir, "transformer.pt")
    scripted.save(model_path)
    return model_path


def test_run_transformer_inference_creates_output_with_prediction_column():
    """Inference should create an MCAP with a populated _predicted column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _make_dummy_scripted_model(tmpdir)
        output_paths = run_transformer_inference(model_path, [TEST_MCAP], data_freq=80)

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_predicted.mcap")
