"""Smoke tests: ScaledModelWrapper must script, save, and load for every model type."""

import os

import torch

from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.torch_model import (
    PlainM5PhysicsModel,
    SpringCoefficientHead,
    SpringTransformerForceEstimator,
    TorchMlpModel,
    TorchRNNModel,
    TorchTransformerModel,
)
from actuator_network.helpers.wrapper import ScaledModelWrapper

DEVICE = torch.device("cpu")


def _script_save_load(wrapped: ScaledModelWrapper, name: str) -> torch.jit.ScriptModule:
    wrapped.freeze()
    save_path = os.path.join(str(wrapped._tmpdir), name)
    wrapped.script_and_save(save_path)
    wrapped.unfreeze()

    loaded = torch.jit.load(save_path, map_location=DEVICE)
    loaded.eval()
    return loaded


def test_wrapper_scripts_mlp(tmp_path):
    torch.manual_seed(0)
    mlp = TorchMlpModel(input_size=2, output_size=1, hidden_layers=[4], device=DEVICE)
    input_mean, input_std = torch.zeros(1, 2), torch.ones(1, 2)
    output_mean, output_std = torch.zeros(1, 1), torch.ones(1, 1)
    wrapped = ScaledModelWrapper(mlp, input_mean, input_std, output_mean, output_std, frequency=80, stride=1)
    wrapped._tmpdir = tmp_path

    loaded = _script_save_load(wrapped, "mlp.pt")

    assert loaded.metadata["frequency"] == 80
    assert loaded.metadata["stride"] == 1
    out = loaded(torch.randn(4, 2))
    assert out.shape == (4, 1)


def test_wrapper_scripts_rnn_with_hidden_state(tmp_path):
    torch.manual_seed(0)
    rnn = TorchRNNModel(input_size=2, hidden_size=4, num_layers=1, output_size=1, device=DEVICE, dropout=0.0)
    input_mean, input_std = torch.zeros(1, 2), torch.ones(1, 2)
    output_mean, output_std = torch.zeros(1, 1), torch.ones(1, 1)
    wrapped = ScaledModelWrapper(
        rnn, input_mean, input_std, output_mean, output_std, frequency=80, history_size=3, stride=1
    )
    wrapped._tmpdir = tmp_path

    loaded = _script_save_load(wrapped, "rnn.pt")

    out = loaded(torch.randn(1, 3, 2))
    assert out.shape == (1, 3, 1)


def test_wrapper_scripts_plain_m5_with_identity_stats(tmp_path):
    torch.manual_seed(0)
    physics = PlainM5PhysicsModel(m5=M5FrictionModel())
    input_mean, input_std = torch.zeros(1, 2), torch.ones(1, 2)
    output_mean, output_std = torch.zeros(1, 1), torch.ones(1, 1)
    wrapped = ScaledModelWrapper(physics, input_mean, input_std, output_mean, output_std, frequency=80, stride=1)
    wrapped._tmpdir = tmp_path

    loaded = _script_save_load(wrapped, "plain_m5.pt")

    out = loaded(torch.randn(3, 2))
    assert out.shape == (3, 1, 4)


def test_wrapper_scripts_spring_transformer_estimator(tmp_path):
    torch.manual_seed(0)
    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=16,
        num_layers=1,
        history_size=8,
        num_heads=2,
        hidden_dim=8,
        device=DEVICE,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + 16,
        output_size=1,
        num_layers=1,
        history_size=4,
        num_heads=2,
        hidden_dim=8,
        device=DEVICE,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=16, device=DEVICE)
    estimator = SpringTransformerForceEstimator(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=16,
        input_mean=torch.zeros(1, 2),
        input_std=torch.ones(1, 2),
        spring_input_mean=torch.zeros(1, 2),
        spring_input_std=torch.ones(1, 2),
        velocity_threshold=0.1,
        spring_alpha=0.9,
        spring_stride=2,
        force_stride=2,
    )
    input_mean, input_std = torch.zeros(1, 2), torch.ones(1, 2)
    output_mean, output_std = torch.zeros(1, 2), torch.ones(1, 2)
    wrapped = ScaledModelWrapper(
        estimator,
        input_mean,
        input_std,
        output_mean,
        output_std,
        frequency=200,
        history_size=4,
        stride=1,
    )
    wrapped._tmpdir = tmp_path

    loaded = _script_save_load(wrapped, "spring.pt")

    assert loaded.metadata["frequency"] == 200
    assert loaded.metadata["history_size"] == 4
    out = loaded(torch.randn(1, 4, 2))
    assert out.shape == (1, 1, 2)
