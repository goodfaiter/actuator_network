"""Tests for M5 friction envelope inference."""

import json
import os
from unittest import mock

import numpy as np
import pandas as pd
import torch

from actuator_network.helpers.hyperparameters import M5FrictionConfig
from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.torch_model import M5EnvelopeFrictionModel
from actuator_network.test_m5 import FRICTION_ENVELOPE_COL, run_m5_inference


def _make_dummy_model(tmpdir: str) -> str:
    """Build and save a tiny M5 envelope model state dict for inference tests."""
    model = M5EnvelopeFrictionModel(device=torch.device("cpu"))
    model.eval()
    model_path = os.path.join(tmpdir, "m5_model.pt")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(tmpdir, "m5_params.json"), "w") as f:
        json.dump({**model.physical_parameters(), "fixed": list(model.fixed_names)}, f)
    return model_path


def _make_fixture_df(n: int = 600) -> pd.DataFrame:
    """Synthetic processed DataFrame covering the columns the pipeline needs."""
    index = pd.to_timedelta(np.arange(n) * 5.0, unit="ms")
    velocity = np.concatenate([np.full(n // 2, 0.5), np.full(n - n // 2, -0.5)])
    return pd.DataFrame(
        {
            "measured_current_amp_data": np.full(n, 0.1),
            "measured_velocity_rad_per_sec_data": velocity,
            "measured_position_rad_data": np.linspace(0.0, 1.0, n),
            "desired_position_rad_data": np.zeros(n),
            "weight_kg_data": np.zeros(n),
            "bota_wrench_N_and_Nm_torque_z": np.full(n, -0.05),
        },
        index=index,
    )


def test_run_m5_inference_creates_output_with_envelope_column(tmp_path):
    """Inference should create an MCAP with a populated, non-negative envelope column."""
    df = _make_fixture_df()
    model_path = _make_dummy_model(str(tmp_path))
    fake_input = str(tmp_path / "fixture.mcap")

    with mock.patch("actuator_network.test_m5.load_mcap_dataframes_parallel_cached", return_value=[df]):
        output_paths = run_m5_inference(model_path, [fake_input], config=M5FrictionConfig())

    assert len(output_paths) == 1
    assert os.path.isfile(output_paths[0])
    assert output_paths[0].endswith("_m5_predicted.mcap")

    # The envelope is a magnitude (softplus-positive parameters), so the
    # written column must be finite and non-negative.
    out_df = read_mcap_to_dataframe(output_paths[0], topics=[f"/{FRICTION_ENVELOPE_COL}"])
    assert FRICTION_ENVELOPE_COL + "_data" in out_df.columns
    assert (out_df[FRICTION_ENVELOPE_COL + "_data"] >= 0).all()
