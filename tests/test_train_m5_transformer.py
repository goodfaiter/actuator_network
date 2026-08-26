import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.torch_model import M5TransformerPhysicsModel, TorchTransformerModel
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper
from actuator_network.train_m5_transformer import load_m5_model, train_m5_transformer


def _dummy_m5_params() -> dict[str, float]:
    """Return deterministic physical parameters for smoke testing."""
    return {
        "K_v": 0.1,
        "K_c": 0.2,
        "K_m": 0.3,
        "K_e": 0.4,
        "V_s": 0.5,
        "alpha": 1.0,
        "K_cs": 0.6,
        "K_ms": 0.7,
        "K_es": 0.8,
    }


def _make_model(input_dim: int = 2, history_size: int = 8, output_dim: int = 1):
    """Create a small M5TransformerPhysicsModel for testing."""
    m5 = M5FrictionModel()
    transformer = TorchTransformerModel(
        input_size=input_dim,
        output_size=output_dim,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=8,
        device=torch.device("cpu"),
    )
    input_mean = torch.zeros(input_dim)
    input_std = torch.ones(input_dim)
    output_mean = torch.zeros(output_dim)
    output_std = torch.ones(output_dim)

    model = M5TransformerPhysicsModel(
        m5=m5,
        transformer=transformer,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        delta_position_idx=0,
        velocity_idx=1,
    )
    return model


def _initial_motor_gain(combined: M5TransformerPhysicsModel) -> float:
    """Return the current motor gain of the combined model as a float."""
    return float(combined.m5.motor_gain_value().item())


def test_m5_transformer_physics_forward_shape():
    """The combined model should return [Batch, 1, 4] physics channels."""
    model = _make_model()
    x = torch.randn(4, 8, 2)
    out = model(x)
    assert out.shape == (4, 1, 4)


def test_m5_transformer_physics_computes_tau_external():
    """With a deterministic M5 (near-constant friction), channel 0 equals tau_motor - tau_friction."""
    m5 = M5FrictionModel()
    # Make M5 return a near-constant friction dominated by K_c = 0.5.
    m5.set_physical_parameters(
        {
            "K_v": 1e-6,
            "K_c": 0.5,
            "K_m": 1e-6,
            "K_e": 1e-6,
            "V_s": 1e6,
            "alpha": 1.0,
            "K_cs": 1e-6,
            "K_ms": 1e-6,
            "K_es": 1e-6,
        }
    )
    with torch.no_grad():
        # Make the Stribeck envelope essentially 1.0 for any velocity.
        m5.V_s_log.fill_(20.0)

    transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=8,
        num_heads=2,
        hidden_dim=8,
        device=torch.device("cpu"),
    )

    input_mean = torch.zeros(2)
    input_std = torch.ones(2)
    output_mean = torch.tensor([0.5])
    output_std = torch.tensor([2.0])

    model = M5TransformerPhysicsModel(
        m5=m5,
        transformer=transformer,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        delta_position_idx=0,
        velocity_idx=1,
    )

    batch = 4
    history = 8
    x = torch.randn(batch, history, 2)
    delta_position = x[:, -1, 0]
    with torch.no_grad():
        motor_gain = model.m5.motor_gain_value().item()
        tau_motor = motor_gain * delta_position
        # Velocity is zero in the constructed input, so the expected friction is
        # approximately K_c plus negligible K_m/K_e terms.
        expected_friction = m5(torch.zeros(batch), tau_motor, torch.zeros(batch))
        expected_phys = tau_motor - expected_friction
        expected_norm = (expected_phys - output_mean) / output_std
        out = model(x)

    assert out.shape == (batch, 1, 4)
    assert torch.allclose(out[:, 0, 0], expected_norm, atol=1e-4)


def test_m5_transformer_physics_gradients_flow_to_transformer():
    """Backpropagation should reach the Transformer parameters through M5."""
    model = _make_model()
    model.m5.requires_grad_(False)

    x = torch.randn(4, 8, 2)
    target = torch.randn(4, 1, 1)

    out = model(x)
    loss = torch.nn.functional.mse_loss(out[:, :, 0:1], target)
    loss.backward()

    transformer_params = list(model.transformer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in transformer_params)


def test_load_m5_model_respects_trainable_flag():
    """Loading with trainable/motor_gain_trainable flags should set requires_grad accordingly."""
    params = _dummy_m5_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        fully_frozen = load_m5_model(params_path, torch.device("cpu"), trainable=False, motor_gain_trainable=False)
        assert not any(p.requires_grad for p in fully_frozen.parameters())
        loaded_frozen = fully_frozen.named_physical_parameters()
        for key, value in params.items():
            assert loaded_frozen[key] == pytest.approx(value, abs=1e-5)

        fully_trainable = load_m5_model(params_path, torch.device("cpu"), trainable=True, motor_gain_trainable=True)
        assert all(p.requires_grad for p in fully_trainable.parameters())

        friction_only = load_m5_model(params_path, torch.device("cpu"), trainable=True, motor_gain_trainable=False)
        assert friction_only.K_v_log.requires_grad
        assert not friction_only.motor_gain_log.requires_grad


def _make_small_combined_model(params_path: str, device: torch.device, trainable: bool):
    """Create a tiny M5TransformerPhysicsModel wrapped for training tests."""
    m5 = load_m5_model(params_path, device, trainable=trainable)
    transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=2,
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
        history_size=2,
        stride=1,
        prediction=False,
        input_columns=["delta_position_rad_data", "measured_velocity_rad_per_sec_data"],
        output_columns=[
            "tendon_bota_force_newton_data",
            "tau_motor_newton_data",
            "tau_friction_newton_data",
            "tau_external_pred_newton_data",
        ],
    )
    return combined, wrapped


def test_train_m5_transformer_updates_m5_when_trainable():
    """Joint training should update M5 parameters when m5_trainable=True."""
    device = torch.device("cpu")
    params = _dummy_m5_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        combined, wrapped = _make_small_combined_model(params_path, device, trainable=True)
        saver = ModelSaver(wrapped, tmpdir)

        inputs = torch.randn(64, 2, 2)
        outputs = torch.randn(64, 1, 1)
        val_inputs = torch.randn(16, 2, 2)
        val_outputs = torch.randn(16, 1, 1)
        initial_kv = float(torch.nn.functional.softplus(combined.m5.K_v_log).item())

        with patch("actuator_network.train_m5_transformer.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            train_m5_transformer(
                combined,
                inputs,
                outputs,
                val_inputs,
                val_outputs,
                model_saver=saver,
                num_epochs=2,
                batch_size=16,
                aux_weight=0.1,
                max_grad_norm=1.0,
            )

        final_kv = float(torch.nn.functional.softplus(combined.m5.K_v_log).item())
        assert final_kv != pytest.approx(initial_kv, abs=1e-6)


def test_train_m5_transformer_keeps_m5_fixed_when_not_trainable():
    """Joint training should not update M5 parameters when m5_trainable=False."""
    device = torch.device("cpu")
    params = _dummy_m5_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        combined, wrapped = _make_small_combined_model(params_path, device, trainable=False)
        saver = ModelSaver(wrapped, tmpdir)

        inputs = torch.randn(64, 2, 2)
        outputs = torch.randn(64, 1, 1)
        val_inputs = torch.randn(16, 2, 2)
        val_outputs = torch.randn(16, 1, 1)
        initial_kv = float(torch.nn.functional.softplus(combined.m5.K_v_log).item())

        with patch("actuator_network.train_m5_transformer.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            train_m5_transformer(
                combined,
                inputs,
                outputs,
                val_inputs,
                val_outputs,
                model_saver=saver,
                num_epochs=2,
                batch_size=16,
                aux_weight=0.1,
                max_grad_norm=1.0,
            )

        final_kv = float(torch.nn.functional.softplus(combined.m5.K_v_log).item())
        assert final_kv == pytest.approx(initial_kv, abs=1e-6)


def test_train_m5_transformer_updates_motor_gain_when_trainable():
    """Joint training should update the M5 motor gain when it is trainable."""
    device = torch.device("cpu")
    params = _dummy_m5_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        combined, wrapped = _make_small_combined_model(params_path, device, trainable=False)
        # Make sure only the motor gain is trainable.
        combined.m5.set_friction_trainable(False)
        combined.m5.motor_gain_log.requires_grad = True
        saver = ModelSaver(wrapped, tmpdir)

        inputs = torch.randn(64, 2, 2)
        outputs = torch.randn(64, 1, 1)
        val_inputs = torch.randn(16, 2, 2)
        val_outputs = torch.randn(16, 1, 1)
        initial_gain = _initial_motor_gain(combined)

        with patch("actuator_network.train_m5_transformer.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            train_m5_transformer(
                combined,
                inputs,
                outputs,
                val_inputs,
                val_outputs,
                model_saver=saver,
                num_epochs=2,
                batch_size=16,
                aux_weight=0.1,
                max_grad_norm=1.0,
            )

        final_gain = _initial_motor_gain(combined)
        assert final_gain != pytest.approx(initial_gain, abs=1e-6)


def test_train_m5_transformer_keeps_motor_gain_fixed_when_not_trainable():
    """Joint training should not update the M5 motor gain when it is frozen."""
    device = torch.device("cpu")
    params = _dummy_m5_params()
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f)

        combined, wrapped = _make_small_combined_model(params_path, device, trainable=False)
        combined.m5.set_friction_trainable(False)
        combined.m5.motor_gain_log.requires_grad = False
        saver = ModelSaver(wrapped, tmpdir)

        inputs = torch.randn(64, 2, 2)
        outputs = torch.randn(64, 1, 1)
        val_inputs = torch.randn(16, 2, 2)
        val_outputs = torch.randn(16, 1, 1)
        initial_gain = _initial_motor_gain(combined)

        with patch("actuator_network.train_m5_transformer.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            train_m5_transformer(
                combined,
                inputs,
                outputs,
                val_inputs,
                val_outputs,
                model_saver=saver,
                num_epochs=2,
                batch_size=16,
                aux_weight=0.1,
                max_grad_norm=1.0,
            )

        final_gain = _initial_motor_gain(combined)
        assert final_gain == pytest.approx(initial_gain, abs=1e-6)
