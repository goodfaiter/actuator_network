"""Tests for the spring transformer training pipeline."""

import pandas as pd
import torch

from actuator_network.helpers.torch_model import (
    SpringCoefficientHead,
    SpringTransformerModel,
    TorchTransformerModel,
)
from actuator_network.helpers.wrapper import ScaledModelWrapper
from actuator_network.train_spring_transformer import (
    _build_aligned_windows,
    _build_frozen_spring_windows,
    build_spring_dataset,
    compute_spring_dataset_stats,
)


def test_build_frozen_spring_windows():
    # Velocity is the second channel.
    normal_windows = torch.zeros(5, 3, 2)
    normal_windows[:, :, 1] = torch.tensor([0.0, 0.0, 0.0])  # all below threshold
    normal_windows[2, -1, 1] = 0.5  # one moving window

    frozen = _build_frozen_spring_windows(normal_windows, velocity_idx=1, threshold_lo=-0.1, threshold_hi=0.1)

    # Before the first moving window, the buffer is zero-initialized.
    assert torch.allclose(frozen[0], torch.zeros_like(normal_windows[0]))
    assert torch.allclose(frozen[1], torch.zeros_like(normal_windows[0]))

    # Moving window is kept as-is.
    assert torch.allclose(frozen[2], normal_windows[2])

    # After the moving window, the buffer is frozen.
    assert torch.allclose(frozen[3], normal_windows[2])
    assert torch.allclose(frozen[4], normal_windows[2])


def test_build_aligned_windows_zero_pads_early_samples():
    """Early samples should be included with zero-padded history windows."""
    data = torch.arange(20, dtype=torch.float32).view(10, 2)  # 10 samples, 2 features
    spring_history_size = 3
    force_history_size = 2
    spring_stride = 2
    force_stride = 1

    spring_windows, force_windows = _build_aligned_windows(
        data,
        spring_history_size=spring_history_size,
        force_history_size=force_history_size,
        spring_stride=spring_stride,
        force_stride=force_stride,
    )

    # We should get one window per sample.
    assert spring_windows.shape == (10, spring_history_size, 2)
    assert force_windows.shape == (10, force_history_size, 2)

    # The first sample has no valid history before it, so only the last entry
    # (the current timestep) is nonzero.
    assert torch.allclose(spring_windows[0, -1], data[0])
    assert torch.allclose(spring_windows[0, :-1], torch.zeros_like(spring_windows[0, :-1]))
    assert torch.allclose(force_windows[0, -1], data[0])
    assert torch.allclose(force_windows[0, :-1], torch.zeros_like(force_windows[0, :-1]))

    # Later samples should contain actual data at the end and zeros at the start.
    # Spring window end timestep for sample i uses data[i] (stride 2, history 3).
    assert torch.allclose(spring_windows[-1, -1], data[-1])
    assert torch.allclose(force_windows[-1, -1], data[-1])


def test_build_spring_dataset_includes_all_samples():
    """The dataset should include every sample with zero-padded early windows."""
    num_samples = 12
    df = pd.DataFrame(
        {
            "measured_position_rad_data": torch.linspace(0, 1, num_samples).tolist(),
            "desired_position_rad_data": torch.linspace(1, 2, num_samples).tolist(),
            "measured_velocity_rad_per_sec_data": torch.linspace(-1, 1, num_samples).tolist(),
            "tendon_bota_force_newton_data": torch.sin(torch.linspace(0, 4 * 3.14159, num_samples)).tolist(),
        }
    )

    stats = compute_spring_dataset_stats(dataframes=[df], file_labels=[("dummy.mcap", 0.5)])

    spring_windows, force_windows, spring_targets, force_targets = build_spring_dataset(
        dataframes=[df],
        file_labels=[("dummy.mcap", 0.5)],
        spring_history_size=4,
        history_size=2,
        spring_stride=2,
        force_stride=1,
        velocity_bounds=(0.0, 0.0),  # every non-zero velocity triggers a spring-buffer update.
        stats=stats,
        device=torch.device("cpu"),
    )

    assert spring_windows.shape[0] == num_samples
    assert force_windows.shape[0] == num_samples
    assert spring_targets.shape[0] == num_samples
    assert force_targets.shape[0] == num_samples

    input_mean, input_std, force_output_mean, force_output_std, spring_output_mean, spring_output_std = stats

    # First windows are zero-padded before the current timestep (in normalized space).
    first_features = torch.tensor(
        df.iloc[0][
            [
                "measured_position_rad_data",
                "desired_position_rad_data",
                "measured_velocity_rad_per_sec_data",
            ]
        ].to_numpy(),
        dtype=torch.float32,
    )
    expected_first = (first_features - input_mean) / input_std
    assert torch.allclose(spring_windows[0, -1], expected_first)
    assert torch.allclose(spring_windows[0, :-1], torch.zeros_like(spring_windows[0, :-1]))
    assert torch.allclose(force_windows[0, -1], expected_first)
    assert torch.allclose(force_windows[0, :-1], torch.zeros_like(force_windows[0, :-1]))

    # Targets line up with the last (current) timestep of each window (normalized).
    expected_force_last = (
        torch.tensor(df["tendon_bota_force_newton_data"].iloc[-1], dtype=torch.float32) - force_output_mean
    ) / force_output_std
    assert torch.allclose(force_targets[-1, 0, 0], expected_force_last)

    expected_spring = (0.5 - spring_output_mean.view(-1)[0]) / spring_output_std.view(-1)[0]
    assert torch.allclose(spring_targets[:, 0, 0], torch.full((num_samples,), float(expected_spring)))


def _make_dummy_stats(device: torch.device, dims: int):
    mean = torch.zeros(1, dims, device=device)
    std = torch.ones(1, dims, device=device)
    return mean, std


def test_spring_force_training_model_forward():
    device = torch.device("cpu")
    spring_history_size = 600
    history_size = 10
    batch_size = 4
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
    )

    # Inputs are already normalized; zero mean / unit std dummy stats make the
    # raw random values valid normalized inputs for this smoke test.
    spring_windows = torch.randn(batch_size, spring_history_size, 2)
    force_windows = torch.randn(batch_size, history_size, 2)
    pred = model(spring_windows, force_windows)

    assert pred.shape == (batch_size, 1, 2)


def test_spring_transformer_force_estimator_stateful():
    device = torch.device("cpu")
    spring_history_size = 600
    history_size = 10
    spring_stride = 2
    force_stride = 2
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=1.0,
        spring_stride=spring_stride,
        force_stride=force_stride,
    )
    model.eval()

    # Static input: spring buffer should freeze and produce identical outputs.
    static_input = torch.zeros(1, 1, 2)
    out1 = model(static_input)
    out2 = model(static_input)
    assert out1.shape == (1, 1, 2)
    assert torch.allclose(out1, out2)

    model.reset()
    out3 = model(static_input)
    assert torch.allclose(out1, out3)


def test_spring_transformer_force_estimator_scriptable():
    device = torch.device("cpu")
    spring_history_size = 600
    history_size = 10
    spring_stride = 2
    force_stride = 2
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_stride=spring_stride,
        force_stride=force_stride,
    )

    scripted = torch.jit.script(model)
    x = torch.zeros(1, 1, 2)
    out = scripted(x)
    assert out.shape == (1, 1, 2)


def test_wrapped_spring_transformer_force_estimator_scriptable():
    device = torch.device("cpu")
    spring_history_size = 600
    history_size = 10
    spring_stride = 2
    force_stride = 2
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)
    spring_out_mean, spring_out_std = _make_dummy_stats(device, 1)
    force_out_mean, force_out_std = _make_dummy_stats(device, 1)

    deployable = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=1.0,
        spring_stride=spring_stride,
        force_stride=force_stride,
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
        stride=force_stride,
        input_columns=["delta_position_rad_data", "measured_velocity_rad_per_sec_data"],
        output_columns=["tendon_bota_force_newton_data", "spring_coeff"],
    )
    wrapped.eval()

    scripted = torch.jit.script(wrapped)
    x = torch.zeros(1, 1, 2)
    out = scripted(x)
    assert out.shape == (1, 1, 2)

    # A freshly scripted model should produce the same output on the same input
    # because its stateful buffers start from the same initial values.
    scripted2 = torch.jit.script(wrapped)
    out2 = scripted2(x)
    assert torch.allclose(out, out2)


def test_spring_transformer_force_estimator_smoothing():
    device = torch.device("cpu")
    spring_history_size = 600
    history_size = 10
    spring_stride = 2
    force_stride = 2
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    # With alpha=0.0 the latent estimate should stay pinned to the initial zero,
    # which makes the spring-coefficient head output constant as well.
    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=0.0,
        spring_stride=spring_stride,
        force_stride=force_stride,
    )
    model.eval()

    static_input = torch.zeros(1, 1, 2)
    out1 = model(static_input)
    out2 = model(static_input)
    assert torch.allclose(model.last_latent, torch.zeros_like(model.last_latent), atol=1e-6)
    assert torch.allclose(out1[0, 0, 1], out2[0, 0, 1], atol=1e-6)


def test_spring_transformer_force_estimator_stride_rate():
    device = torch.device("cpu")
    spring_history_size = 10
    history_size = 5
    spring_stride = 4
    force_stride = 2
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=1.0,
        spring_stride=spring_stride,
        force_stride=force_stride,
    )

    # Two different moving inputs. Dummy stats are zero mean / unit std, so
    # normalized values equal raw values.
    input_a = torch.zeros(1, 1, 2)
    input_a[0, -1, 1] = 1.0  # velocity above threshold
    input_b = torch.ones(1, 1, 2)
    input_b[0, -1, 1] = 1.0  # velocity above threshold

    # Call 0 is a spring sample: buffer should update to input_a.
    _ = model(input_a)
    assert torch.allclose(model.spring_buffer[0, -1, :], input_a[0, -1, :])

    # Call 1 is not a spring sample: buffer should stay as input_a.
    _ = model(input_b)
    assert torch.allclose(model.spring_buffer[0, -1, :], input_a[0, -1, :])

    # Call 2 is a spring sample again: buffer should update to input_b.
    _ = model(input_b)
    assert torch.allclose(model.spring_buffer[0, -1, :], input_b[0, -1, :])


def test_spring_transformer_force_estimator_negative_velocity_updates_buffer():
    """A negative velocity with magnitude above the threshold should also update the buffer."""
    device = torch.device("cpu")
    spring_history_size = 10
    history_size = 5
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    in_mean, in_std = _make_dummy_stats(device, 2)
    spring_in_mean, spring_in_std = _make_dummy_stats(device, 2)

    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=1.0,
        spring_stride=2,
        force_stride=2,
    )

    input_negative = torch.zeros(1, 1, 2)
    input_negative[0, -1, 1] = -1.0  # negative velocity with magnitude above threshold

    _ = model(input_negative)
    assert torch.allclose(model.spring_buffer[0, -1, :], input_negative[0, -1, :])


def test_transformer_default_activation_is_relu():
    device = torch.device("cpu")
    model = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=10,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    x = torch.randn(2, 10, 2)
    out = model(x)
    assert out.shape == (2, 1, 1)


def test_transformer_supports_different_activations():
    device = torch.device("cpu")
    for activation in ["relu", "tanh", "gelu", "leaky_relu", "elu", "silu"]:
        model = TorchTransformerModel(
            input_size=2,
            output_size=1,
            num_layers=1,
            history_size=10,
            num_heads=2,
            hidden_dim=16,
            device=device,
            activation=activation,
        )
        x = torch.randn(2, 10, 2)
        out = model(x)
        assert out.shape == (2, 1, 1), f"Unexpected output shape for {activation}"


def test_transformer_raises_on_invalid_activation():
    device = torch.device("cpu")
    try:
        TorchTransformerModel(
            input_size=2,
            output_size=1,
            num_layers=1,
            history_size=10,
            num_heads=2,
            hidden_dim=16,
            device=device,
            activation="not_an_activation",
        )
    except ValueError as e:
        assert "Unsupported activation" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid activation")


def test_spring_transformer_force_estimator_multi_env():
    device = torch.device("cpu")
    spring_history_size = 10
    history_size = 5
    latent_dim = 16

    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=latent_dim,
        num_layers=1,
        history_size=spring_history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + latent_dim,
        output_size=1,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=latent_dim, device=device)

    model = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=latent_dim,
        velocity_idx=1,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=1.0,
        spring_stride=2,
        force_stride=2,
    )
    model.eval()

    # Env 0 moves (velocity 1.0), env 1 stands still (velocity 0).
    moving_input = torch.zeros(2, 1, 2)
    moving_input[0, 0, 1] = 1.0

    _ = model(moving_input)
    _ = model(moving_input)

    # Env 0's spring buffer holds the moving sample; env 1's stays zero-frozen.
    assert torch.allclose(model.spring_buffer[0, -1, :], moving_input[0, 0, :])
    assert torch.allclose(model.spring_buffer[1], torch.zeros_like(model.spring_buffer[1]))

    # The force buffers advance for both environments.
    assert torch.allclose(model.force_buffer[0, -1, :], moving_input[0, 0, :])
    assert torch.allclose(model.force_buffer[1, -1, :], moving_input[1, 0, :])
    assert int(model.spring_update_counter[0].item()) == 2
    assert int(model.spring_update_counter[1].item()) == 2

    # Once env 1 also moves, its spring buffer updates.
    _ = model(moving_input)
    assert torch.allclose(model.spring_buffer[1, -1, :], moving_input[1, 0, :])

    # Selectively reset environment 0 only; env 1 keeps its state.
    model.reset(torch.tensor([True, False]))
    assert torch.allclose(model.spring_buffer[0], torch.zeros_like(model.spring_buffer[0]))
    assert torch.allclose(model.force_buffer[0], torch.zeros_like(model.force_buffer[0]))
    assert int(model.spring_update_counter[0].item()) == 0
    assert torch.allclose(model.spring_buffer[1, -1, :], moving_input[1, 0, :])
    assert int(model.spring_update_counter[1].item()) == 3


def _make_stateful_spring_model(device: torch.device, seed: int = 0, **overrides) -> SpringTransformerModel:
    """Create a small stateful spring transformer for the stateful tests below."""
    torch.manual_seed(seed)
    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=16,
        num_layers=1,
        history_size=8,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + 16,
        output_size=1,
        num_layers=1,
        history_size=4,
        num_heads=2,
        hidden_dim=16,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=16, device=device)
    params: dict = dict(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=16,
        velocity_idx=1,
        velocity_threshold_lo=-0.1,
        velocity_threshold_hi=0.1,
        spring_alpha=0.3,
        spring_stride=4,
        force_stride=2,
    )
    params.update(overrides)
    return SpringTransformerModel(**params)


def _make_stateful_tick(x_pos: float, v0: float, v1: float) -> torch.Tensor:
    """Build a two-env online input tick with the velocity channel indexed 1."""
    x = torch.zeros(2, 1, 2)
    x[:, 0, 0] = x_pos
    x[0, 0, 1] = v0
    x[1, 0, 1] = v1
    return x


def _reference_stateful_tick(
    model: SpringTransformerModel,
    x: torch.Tensor,
    state: dict,
) -> torch.Tensor:
    """Replicate the pre-optimization ``_forward_stateful`` semantics.

    The reference keeps its own state (separate from the model buffers) and
    calls the model's stateless sub-modules, so outputs must match the
    optimized model tick for tick.
    """
    spring_buffer = state["spring_buffer"]
    force_buffer = state["force_buffer"]
    last_latent = state["last_latent"]

    last_velocity = x[:, -1, model.velocity_idx]
    is_spring_sample = (state["counter"] % model.spring_update_ratio) == 0
    moving = (last_velocity > model.velocity_threshold_hi) | (last_velocity < model.velocity_threshold_lo)
    spring_mask = is_spring_sample & moving

    if bool(spring_mask.any()):
        shift = torch.cat([spring_buffer[spring_mask][:, 1:, :], x[spring_mask][:, -1:, :]], dim=1)
        spring_buffer[spring_mask] = shift

    force_shifted = torch.cat([force_buffer[:, 1:, :], x[:, -1:, :]], dim=1)
    force_buffer.copy_(force_shifted)

    state["counter"] += 1

    latent_norm = model.model_transformer(spring_buffer)
    smoothed_latent = model.spring_alpha * latent_norm + (1.0 - model.spring_alpha) * last_latent
    last_latent.copy_(smoothed_latent)

    latent_channel = smoothed_latent.expand(-1, force_buffer.size(1), -1)
    force_input_norm = torch.cat([force_buffer, latent_channel], dim=-1)
    force_pred_norm = model.force_transformer(force_input_norm)
    spring_pred_norm = model.spring_coeff_head(smoothed_latent)

    return torch.cat([force_pred_norm, spring_pred_norm], dim=-1)


def _reference_reset_state(state: dict, reset_idx: torch.Tensor | None = None) -> None:
    """Mirror ``SpringTransformerModel.reset`` on the reference state."""
    if reset_idx is None:
        state["spring_buffer"].zero_()
        state["force_buffer"].zero_()
        state["last_latent"].zero_()
        state["counter"].zero_()
        return
    mask = reset_idx.reshape(-1)
    state["spring_buffer"][mask] = 0.0
    state["force_buffer"][mask] = 0.0
    state["last_latent"][mask] = 0.0
    state["counter"][mask] = 0


def _reference_state_dict(model: SpringTransformerModel, num_envs: int) -> dict:
    """Initialize the reference state with the model's fresh (zero) per-env state.

    The model's registered buffers hold a single environment until the first
    forward call resizes them, so the reference cannot just copy them.
    """
    device = next(model.parameters()).device
    input_size = model.model_transformer.input_size
    return dict(
        spring_buffer=torch.zeros(num_envs, model.spring_history_size, input_size, device=device),
        force_buffer=torch.zeros(num_envs, model.force_history_size, input_size, device=device),
        last_latent=torch.zeros(num_envs, 1, model.latent_dim, device=device),
        counter=torch.zeros(num_envs, dtype=torch.int64, device=device),
    )


def test_stateful_frozen_skip_equivalence():
    """The optimized stateful path must match the original semantics tick for tick.

    Drives moving, frozen, fresh, and selectively-reset environments and
    compares against an inline reference of the original semantics.
    """
    device = torch.device("cpu")
    model = _make_stateful_spring_model(device, seed=0)
    model.eval()

    # The default model has spring_stride=4, force_stride=2 (every other tick
    # is a spring sample tick).
    ticks = [
        _make_stateful_tick(0.0, 0.0, 0.0),  # both frozen: transformer runs once (fresh state)
        _make_stateful_tick(0.1, 0.0, 0.5),  # env 1 moving but no spring sample: buffers frozen
        _make_stateful_tick(0.2, 0.3, 0.5),  # spring sample: both update
        _make_stateful_tick(0.3, 0.0, 0.0),  # both frozen
        _make_stateful_tick(0.4, -0.5, 0.0),  # spring sample: env 0 updates with negative velocity
        _make_stateful_tick(0.5, 0.0, 0.0),  # both frozen
        _make_stateful_tick(0.6, 2.0, -2.0),  # spring sample: both update
        _make_stateful_tick(0.7, 0.0, 0.0),  # both frozen
    ]

    model.reset()
    state = _reference_state_dict(model, num_envs=2)

    for i, x in enumerate(ticks):
        if i == 5:
            # Selectively reset env 0 mid-run; env 1 keeps its state.
            reset_mask = torch.tensor([True, False])
            model.reset(reset_mask)
            _reference_reset_state(state, reset_mask)
        out = model(x)
        out_ref = _reference_stateful_tick(model, x, state)
        assert torch.allclose(out, out_ref, rtol=1e-5, atol=1e-6), (
            f"tick {i}: max diff {(out - out_ref).abs().max().item()}"
        )


def test_stateful_frozen_skip_reduces_transformer_calls():
    """All-frozen ticks skip the spring transformer without changing outputs."""
    device = torch.device("cpu")
    torch.manual_seed(2)
    model = _make_stateful_spring_model(device, seed=2)
    model.eval()

    calls = {"count": 0}
    original_forward = model.model_transformer.forward

    def counting_forward(x: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return original_forward(x)

    model.model_transformer.forward = counting_forward

    still = _make_stateful_tick(0.0, 0.0, 0.0)
    still = still[:1]
    moving = _make_stateful_tick(0.0, 1.0, 0.0)
    moving = moving[:1]

    # Tick order (spring sample every other tick): fresh run, frozen, moving run,
    # frozen, moving run, then frozen ticks served from the cached latent.
    ticks = [still, still, moving, still, moving, still, still, still]
    for x in ticks:
        model(x)
    assert calls["count"] == 3


def test_stateful_frozen_skip_scripted_matches_eager():
    """The scripted optimized model matches eager outputs for identical state."""
    device = torch.device("cpu")
    model = _make_stateful_spring_model(device, seed=3)
    model.eval()

    ticks = [
        _make_stateful_tick(0.0, 0.0, 0.0),
        _make_stateful_tick(0.1, 0.0, 0.0),
        _make_stateful_tick(0.2, 1.0, 1.0),
        _make_stateful_tick(0.3, 1.0, 1.0),  # moving but not a spring sample tick
        _make_stateful_tick(0.4, 0.0, 0.0),
        _make_stateful_tick(0.5, 1.0, -1.0),
        _make_stateful_tick(0.6, 0.0, 0.0),
    ]

    model.reset()
    expected = [model(x) for x in ticks]

    scripted = torch.jit.script(model)
    scripted.reset()
    for i, (x, ref) in enumerate(zip(ticks, expected)):
        out = scripted(x)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), f"tick {i}: max diff {(out - ref).abs().max().item()}"

    # A batch change re-initializes the per-env state in the scripted copy.
    single = _make_stateful_tick(0.0, 0.0, 0.0)[:1]
    wide = torch.cat([single, single, single], dim=0)  # 3 envs
    wide[0, 0, 1] = 1.0
    out = scripted(wide)
    assert out.shape == (3, 1, 2)
