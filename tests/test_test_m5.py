import json
import os
import tempfile

import pytest
import torch

from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.test_m5 import load_m5_model, run_m5_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _dummy_params() -> dict[str, float]:
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


def test_load_m5_model_restores_physical_parameters():
    """Loading from JSON should restore the same physical parameters."""
    params = _dummy_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        device = torch.device("cpu")
        model = load_m5_model(params_path, device)

        assert isinstance(model, M5FrictionModel)
        loaded_params = model.named_physical_parameters()
        for key, value in params.items():
            assert loaded_params[key] == pytest.approx(value, abs=1e-5)


def test_run_m5_inference_creates_output_with_prediction_column():
    """Inference should create an MCAP with a populated m5_newton_predicted column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    params = _dummy_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        output_paths = run_m5_inference(params_path, [TEST_MCAP], data_freq=80)

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
