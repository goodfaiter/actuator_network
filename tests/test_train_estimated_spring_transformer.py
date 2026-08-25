"""Tests for the estimated-spring transformer training pipeline."""

import torch

from actuator_network.helpers.torch_model import (
    SpringForceTrainingModel,
    SpringTransformerForceEstimator,
    TorchTransformerModel,
)
from actuator_network.helpers.wrapper import ScaledModelWrapper
from actuator_network.train_estimated_spring_transformer import _build_frozen_spring_windows


def test_build_frozen_spring_windows():
    # Velocity is the second channel.
    normal_windows = torch.zeros(5, 3, 2)
    normal_windows[:, :, 1] = torch.tensor([0.0, 0.0, 0.0])  # all below threshold
    normal_windows[2, -1, 1] = 0.5  # one moving window

    frozen = _build_frozen_spring_windows(normal_windows, velocity_idx=1, velocity_threshold=0.1)

    # Before the first moving window, the buffer is zero-initialized.
    assert torch.allclose(frozen[0], torch.zeros_like(normal_windows[0]))
    assert torch.allclose(frozen[1], torch.zeros_like(normal_windows[0]))

    # Moving window is kept as-is.
    assert torch.allclose(frozen[2], normal_windows[2])

    # After the moving window, the buffer is frozen.
    assert torch.allclose(frozen[3], normal_windows[2])
    assert torch.allclose(frozen[4], normal_windows[2])


def _make_dummy_stats(device: torch.device, dims: int):
    mean = torch.zeros(1, dims, device=device)
    std = torch.ones(1, dims, device=device)
    return mean, std


def test_spring_force_training_model_forward():
    device = torch.device("cpu")
    history_size = 10
    batch_size = 4

    spring_transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )

    model = SpringForceTrainingModel(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
    )

    # Inputs are already normalized; zero mean / unit std dummy stats make the
    # raw random values valid normalized inputs for this smoke test.
    spring_windows = torch.randn(batch_size, history_size, 2)
    force_windows = torch.randn(batch_size, history_size, 2)
    combined_input = torch.stack([spring_windows, force_windows], dim=1)
    pred = model(combined_input)

    assert pred.shape == (batch_size, 1, 2)


def test_spring_transformer_force_estimator_stateful():
    device = torch.device("cpu")
    history_size = 10

    spring_transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    model = SpringTransformerForceEstimator(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
        input_mean=in_mean,
        input_std=in_std,
        spring_input_mean=spring_in_mean,
        spring_input_std=spring_in_std,
        velocity_threshold=0.1,
        spring_alpha=1.0,
    )
    model.eval()

    # Static input: spring buffer should freeze and produce identical outputs.
    static_input = torch.zeros(1, history_size, 2)
    out1 = model(static_input)
    out2 = model(static_input)
    assert out1.shape == (1, 1, 2)
    assert torch.allclose(out1, out2)

    model.reset()
    out3 = model(static_input)
    assert torch.allclose(out1, out3)


def test_spring_transformer_force_estimator_scriptable():
    device = torch.device("cpu")
    history_size = 10

    spring_transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    model = SpringTransformerForceEstimator(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
        input_mean=in_mean,
        input_std=in_std,
        spring_input_mean=spring_in_mean,
        spring_input_std=spring_in_std,
        velocity_threshold=0.1,
    )

    scripted = torch.jit.script(model)
    x = torch.zeros(1, history_size, 2)
    out = scripted(x)
    assert out.shape == (1, 1, 2)


def test_wrapped_spring_transformer_force_estimator_scriptable():
    device = torch.device("cpu")
    history_size = 10

    spring_transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)
    spring_out_mean, spring_out_std = _make_dummy_stats(device, 1)
    force_out_mean, force_out_std = _make_dummy_stats(device, 1)

    deployable = SpringTransformerForceEstimator(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
        input_mean=in_mean,
        input_std=in_std,
        spring_input_mean=spring_in_mean,
        spring_input_std=spring_in_std,
        velocity_threshold=0.1,
        spring_alpha=1.0,
    )

    combined_output_mean = torch.cat([force_out_mean, spring_out_mean], dim=-1)
    combined_output_std = torch.cat([force_out_std, spring_out_std], dim=-1)

    wrapped = ScaledModelWrapper(
        deployable,
        in_mean,
        in_std,
        combined_output_mean,
        combined_output_std,
        frequency=100,
        history_size=history_size,
        stride=2,
        prediction=False,
        input_columns=["delta_position_rad_data", "measured_velocity_rad_per_sec_data"],
        output_columns=["tendon_bota_force_newton_data", "spring_coeff"],
    )
    wrapped.eval()

    scripted = torch.jit.script(wrapped)
    x = torch.zeros(1, history_size, 2)
    out = scripted(x)
    assert out.shape == (1, 1, 2)

    # A freshly scripted model should produce the same output on the same input
    # because its stateful buffers start from the same initial values.
    scripted2 = torch.jit.script(wrapped)
    out2 = scripted2(x)
    assert torch.allclose(out, out2)


def test_spring_transformer_force_estimator_smoothing():
    device = torch.device("cpu")
    history_size = 10

    spring_transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=3,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    # With alpha=0.0 the spring estimate should stay pinned to the initial zero.
    model = SpringTransformerForceEstimator(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
        input_mean=in_mean,
        input_std=in_std,
        spring_input_mean=spring_in_mean,
        spring_input_std=spring_in_std,
        velocity_threshold=0.1,
        spring_alpha=0.0,
    )
    model.eval()

    static_input = torch.zeros(1, history_size, 2)
    out = model(static_input)
    assert torch.allclose(out[0, 0, 1], torch.tensor(0.0), atol=1e-6)
