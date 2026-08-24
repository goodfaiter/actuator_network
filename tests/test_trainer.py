import os
import tempfile
from unittest.mock import MagicMock, patch

import torch

from actuator_network.helpers.torch_model import TorchMlpModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


def _make_wrapped_model() -> ScaledModelWrapper:
    """Create a tiny wrapped MLP for quick training tests."""
    device = torch.device("cpu")
    model = TorchMlpModel(input_size=2, output_size=1, hidden_layers=[4], device=device)
    return ScaledModelWrapper(
        model,
        input_mean=torch.zeros(2),
        input_std=torch.ones(2),
        output_mean=torch.zeros(1),
        output_std=torch.ones(1),
    )


def test_train_default_latest_checkpoints():
    """Default training should produce best_latest.pt and final_latest.pt."""
    wrapped = _make_wrapped_model()
    inputs = torch.randn(64, 2)
    outputs = torch.randn(64, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("actuator_network.helpers.trainer.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            saver = ModelSaver(wrapped, tmpdir)
            train(wrapped, inputs, outputs, model_saver=saver)

        assert os.path.isfile(os.path.join(tmpdir, "best_latest.pt"))
        assert os.path.isfile(os.path.join(tmpdir, "final_latest.pt"))


def test_train_prefixed_latest_checkpoints():
    """Training with a prefix should produce best_<prefix>latest.pt and final_<prefix>latest.pt."""
    wrapped = _make_wrapped_model()
    inputs = torch.randn(64, 2)
    outputs = torch.randn(64, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("actuator_network.helpers.trainer.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            saver = ModelSaver(wrapped, tmpdir)
            train(wrapped, inputs, outputs, model_saver=saver, latest_prefix="m5_transformer_")

        assert os.path.isfile(os.path.join(tmpdir, "best_m5_transformer_latest.pt"))
        assert os.path.isfile(os.path.join(tmpdir, "final_m5_transformer_latest.pt"))
