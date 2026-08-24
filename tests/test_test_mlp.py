import os
import tempfile

import torch

from actuator_network.helpers.torch_model import TorchMlpModel
from actuator_network.helpers.wrapper import ScaledModelWrapper
from actuator_network.test_mlp import run_mlp_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _make_dummy_scripted_model(tmpdir: str) -> str:
    """Build and save a tiny scripted MLP wrapper for inference tests."""
    device = torch.device("cpu")

    # History size of 1 keeps the input dimension small for the test MCAP.
    history_size = 1
    mlp = TorchMlpModel(
        input_size=2 * history_size,
        output_size=1,
        hidden_layers=[4],
        device=device,
    )

    input_mean = torch.zeros(1, 2)
    input_std = torch.ones(1, 2)
    output_mean = torch.zeros(1, 1)
    output_std = torch.ones(1, 1)

    wrapped = ScaledModelWrapper(
        mlp,
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
    mlp.eval()

    scripted = torch.jit.script(wrapped)
    model_path = os.path.join(tmpdir, "mlp.pt")
    scripted.save(model_path)
    return model_path


def test_run_mlp_inference_creates_output_with_prediction_column():
    """Inference should create an MCAP with a populated _predicted column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _make_dummy_scripted_model(tmpdir)
        output_paths = run_mlp_inference(model_path, [TEST_MCAP], data_freq=80)

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_mlp_predicted.mcap")
