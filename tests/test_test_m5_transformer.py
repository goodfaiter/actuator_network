import os
import tempfile

import torch

from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.torch_model import M5TransformerPhysicsModel, TorchTransformerModel
from actuator_network.helpers.wrapper import ScaledModelWrapper
from actuator_network.test_m5_transformer import run_m5_transformer_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _dummy_m5_params() -> dict[str, float]:
    """Return deterministic physical parameters for smoke testing."""
    return {
        "Kv": 0.1,
        "Kc": 0.2,
        "Km": 0.3,
        "Ke": 0.4,
        "V_s": 0.5,
        "alpha": 1.0,
        "Kcs": 0.6,
        "K_ms": 0.7,
        "Kes": 0.8,
    }


def _make_dummy_scripted_model(tmpdir: str) -> str:
    """Build and save a tiny scripted M5 + Transformer wrapper for inference tests."""
    device = torch.device("cpu")

    m5 = M5FrictionModel().to(device)
    m5.set_physical_parameters(_dummy_m5_params())
    m5.eval()
    m5.requires_grad_(False)

    # Small history size so the test MCAP is guaranteed to produce windows.
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

    combined = M5TransformerPhysicsModel(
        m5=m5,
        transformer=transformer,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        delta_position_idx=0,
        velocity_idx=1,
    )

    wrapped = ScaledModelWrapper(
        combined,
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
    combined.eval()

    scripted = torch.jit.script(wrapped)
    model_path = os.path.join(tmpdir, "m5_transformer.pt")
    scripted.save(model_path)
    return model_path


def test_run_m5_transformer_inference_creates_output_with_prediction_column():
    """Inference should create an MCAP with a populated _predicted column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _make_dummy_scripted_model(tmpdir)
        output_paths = run_m5_transformer_inference(model_path, [TEST_MCAP], data_freq=80)

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_m5_transformer_predicted.mcap")
