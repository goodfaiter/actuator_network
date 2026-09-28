import os
import tempfile

import torch

from actuator_network.helpers.hyperparameters import M5FrictionConfig
from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.torch_model import M5EnvelopeFrictionModel
from actuator_network.test_m5 import FRICTION_ENVELOPE_COL, run_m5_inference

TEST_MCAP = "/workspace/tests/test.mcap"


def _make_dummy_model(tmpdir: str) -> str:
    """Build and save a tiny M5 envelope model state dict for inference tests."""
    model = M5EnvelopeFrictionModel(device=torch.device("cpu"))
    model.eval()
    model_path = os.path.join(tmpdir, "m5_model.pt")
    torch.save(model.state_dict(), model_path)
    return model_path


def test_run_m5_inference_creates_output_with_envelope_column():
    """Inference should create an MCAP with a populated, non-negative envelope column."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _make_dummy_model(tmpdir)
        output_paths = run_m5_inference(model_path, [TEST_MCAP], config=M5FrictionConfig())

        assert len(output_paths) == 1
        assert os.path.isfile(output_paths[0])
        assert output_paths[0].endswith("_m5_predicted.mcap")

        # The envelope is a magnitude (softplus-positive parameters), so the
        # written column must be finite and non-negative.
        df = read_mcap_to_dataframe(output_paths[0], topics=[f"/{FRICTION_ENVELOPE_COL}"])
        assert FRICTION_ENVELOPE_COL + "_data" in df.columns
        assert (df[FRICTION_ENVELOPE_COL + "_data"] >= 0).all()
