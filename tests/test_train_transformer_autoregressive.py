from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from actuator_network.helpers.torch_model import TorchTransformerModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver
from actuator_network.train_transformer_autoregressive import build_autoregressive_dataset


def _make_synthetic_dataframe(num_samples: int = 100) -> pd.DataFrame:
    """Create a small deterministic DataFrame with the expected columns."""
    return pd.DataFrame(
        {
            "delta_position_rad_data": torch.linspace(0, 1, num_samples).tolist(),
            "measured_velocity_rad_per_sec_data": torch.linspace(-1, 1, num_samples).tolist(),
            "tendon_bota_force_newton_data": torch.sin(torch.linspace(0, 4 * 3.14159, num_samples)).tolist(),
        }
    )


def test_build_autoregressive_dataset_shifts_force_channel():
    """The force input channel should be shifted by one relative to the output."""
    df = pd.DataFrame(
        {
            "delta_position_rad_data": [0.0] * 5,
            "measured_velocity_rad_per_sec_data": [0.0] * 5,
            "tendon_bota_force_newton_data": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    inputs, outputs = build_autoregressive_dataset(
        [df],
        input_cols=["delta_position_rad_data", "measured_velocity_rad_per_sec_data", "tendon_bota_force_newton_data"],
        output_cols=["tendon_bota_force_newton_data"],
        history_size=3,
        stride=1,
        prediction=False,
        device=torch.device("cpu"),
    )

    # With 5 samples, history_size=3 and stride=1 we get 3 sequences.
    assert inputs.shape == (3, 3, 3)
    assert outputs.shape == (3, 1, 1)
    assert torch.allclose(outputs[:, 0, 0], torch.tensor([3.0, 4.0, 5.0]))
    # Last input timestep's force should be the output at the previous timestep.
    assert torch.allclose(inputs[:, -1, 2], torch.tensor([2.0, 3.0, 4.0]))
    # First input timestep's force should be zero (padding).
    assert torch.allclose(inputs[:, 0, 2], torch.tensor([0.0, 1.0, 2.0]))


def test_train_transformer_autoregressive_smoke():
    """A small training run should complete without errors."""
    device = torch.device("cpu")
    df = _make_synthetic_dataframe(num_samples=200)

    inputs, outputs = build_autoregressive_dataset(
        [df],
        input_cols=["delta_position_rad_data", "measured_velocity_rad_per_sec_data", "tendon_bota_force_newton_data"],
        output_cols=["tendon_bota_force_newton_data"],
        history_size=8,
        stride=1,
        prediction=False,
        device=device,
    )

    # Normalize using training statistics.
    mean = inputs.mean(dim=[0, 1], keepdim=True)
    std = inputs.std(dim=[0, 1], keepdim=True) + 1e-8
    inputs_norm = (inputs - mean) / std

    out_mean = outputs.mean(dim=[0, 1], keepdim=True)
    out_std = outputs.std(dim=[0, 1], keepdim=True) + 1e-8
    outputs_norm = (outputs - out_mean) / out_std

    # Use a small held-out slice of the same synthetic data as validation.
    val_size = 40
    val_inputs_norm = inputs_norm[-val_size:]
    val_outputs_norm = outputs_norm[-val_size:]
    inputs_norm = inputs_norm[:-val_size]
    outputs_norm = outputs_norm[:-val_size]

    model = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=8,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )

    wrapped = ModelSaver  # placeholder to satisfy type checking in the next line
    wrapped = type(
        "ScaledModelWrapper",
        (),
        {
            "model": model,
            "input_mean": mean.squeeze(0),
            "input_std": std.squeeze(0),
            "output_mean": out_mean.squeeze(0),
            "output_std": out_std.squeeze(0),
            "frequency": torch.tensor(200, dtype=torch.int32),
            "history_size": torch.tensor(8, dtype=torch.int32),
            "stride": torch.tensor(1, dtype=torch.int32),
            "prediction_mode": torch.tensor(False),
            "input_columns": [
                "delta_position_rad_data",
                "measured_velocity_rad_per_sec_data",
                "tendon_bota_force_newton_data",
            ],
            "output_columns": ["tendon_bota_force_newton_data"],
            "reset": lambda self: None,
            "forward": lambda self, x: (x - self.input_mean) / self.input_std,
            "freeze": lambda self: None,
            "unfreeze": lambda self: None,
            "trace_and_save": lambda self, path: None,
        },
    )()

    saver = ModelSaver.__new__(ModelSaver)
    saver._wrapped_model = wrapped
    saver._root_folder = "/tmp"
    saver._folder = "/tmp/"
    saver._file_prefix = "/tmp/test_"

    with patch("actuator_network.helpers.trainer.wandb") as mock_wandb:
        mock_wandb.init.return_value = MagicMock()
        train(
            model,
            inputs_norm,
            outputs_norm,
            val_inputs_norm,
            val_outputs_norm,
            model_saver=saver,
            latest_prefix="transformer_autoregressive_",
        )
