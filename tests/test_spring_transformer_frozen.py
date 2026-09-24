"""Tests for the frozen-latent pruned deployment model of the spring transformer."""

import os

import torch

from actuator_network.export_frozen_latent import build_frozen_latent_model, compute_frozen_latent
from actuator_network.helpers.torch_model import (
    FrozenLatentForceModel,
    SpringCoefficientHead,
    SpringTransformerModel,
    TorchTransformerModel,
)
from actuator_network.helpers.wrapper import ScaledModelWrapper

TEST_MCAP = "/workspace/tests/test.mcap"


def _make_spring_model(device: torch.device, seed: int = 0, **overrides) -> SpringTransformerModel:
    """Create a small stateful spring transformer (source of pruned submodules)."""
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


def _make_pruned_model(
    full: SpringTransformerModel,
    frozen_latent: torch.Tensor,
) -> FrozenLatentForceModel:
    """Build the pruned deployment model sharing the full model's submodules."""
    pruned = FrozenLatentForceModel(
        force_transformer=full.force_transformer,
        spring_coeff_head=full.spring_coeff_head,
        latent_dim=16,
        input_size=2,
        force_history_size=4,
    )
    pruned.set_frozen_latent(frozen_latent)
    return pruned


def _tick(x_pos: float, v: float) -> torch.Tensor:
    """Build a single-env online input tick (velocity channel indexed 1)."""
    x = torch.zeros(1, 1, 2)
    x[0, 0, 0] = x_pos
    x[0, 0, 1] = v
    return x


def _frozen_reference(
    estimator: torch.nn.Module,
    x: torch.Tensor,
    frozen_latent: torch.Tensor,
    state: dict,
) -> torch.Tensor:
    """Replicate the frozen-latent tick semantics with an external state.

    ``state["force_buffer"]`` is created lazily with zero-initialization
    (matching the model's fresh state) and advances with each call.
    """
    if state["force_buffer"] is None:
        state["force_buffer"] = torch.zeros(1, int(estimator.force_history_size), int(estimator.input_size))
    state["force_buffer"] = torch.cat([state["force_buffer"][:, 1:, :], x[:, -1:, :]], dim=1)
    latent_channel = frozen_latent.expand(-1, state["force_buffer"].size(1), -1)
    force_input = torch.cat([state["force_buffer"], latent_channel], dim=-1)
    force_pred = estimator.force_transformer(force_input)
    spring_pred = estimator.spring_coeff_head(frozen_latent)
    return torch.cat([force_pred, spring_pred], dim=-1)


def test_frozen_pruned_model_matches_manual_reference():
    device = torch.device("cpu")
    torch.manual_seed(11)
    full = _make_spring_model(device, seed=11)
    full.eval()
    frozen_latent = torch.randn(1, 1, 16)

    pruned = _make_pruned_model(full, frozen_latent)
    pruned.eval()

    ticks = [_tick(0.0, 0.0), _tick(0.2, 1.0), _tick(0.4, -1.0), _tick(0.6, 1.0)]
    state = {"force_buffer": None}
    for i, x in enumerate(ticks):
        out = pruned(x)
        ref = _frozen_reference(pruned, x, frozen_latent, state)
        assert out.shape == (1, 1, 2)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), f"tick {i}: max diff {(out - ref).abs().max().item()}"

    # Per-env semantics: a 2-env call matches fresh single-env runs.
    x = torch.zeros(2, 1, 2)
    x[0, 0, 0] = 0.1
    x[0, 0, 1] = 1.0
    x[1, 0, 0] = -5.0
    out_wide = pruned(x)
    for env_idx in range(2):
        single = _make_pruned_model(full, frozen_latent)
        single.eval()
        single_out = single(x[env_idx : env_idx + 1].clone())
        assert torch.allclose(out_wide[env_idx : env_idx + 1], single_out, rtol=1e-5, atol=1e-6)


def test_frozen_pruned_reset_and_batch_change():
    device = torch.device("cpu")
    torch.manual_seed(12)
    full = _make_spring_model(device, seed=12)
    full.eval()
    frozen_latent = torch.randn(1, 1, 16)
    pruned = _make_pruned_model(full, frozen_latent)
    pruned.eval()

    first = pruned(_tick(0.0, 0.0))
    pruned.reset()
    again = pruned(_tick(0.0, 0.0))
    assert torch.allclose(first, again, atol=1e-6)
    assert torch.allclose(pruned.frozen_latent, frozen_latent), "reset must preserve the latent"

    # Retargeting the latent changes the output.
    new_latent = torch.randn(1, 1, 16)
    pruned.set_frozen_latent(new_latent)
    out_new = pruned(_tick(0.0, 0.0))
    assert not torch.allclose(out_new, first, atol=1e-5)

    # A batch change keeps the latent but gives a fresh force buffer.
    wide = torch.cat([_tick(0.0, 0.0), _tick(0.3, 1.0)], dim=0)
    out_wide = pruned(wide)
    assert out_wide.shape == (2, 1, 2)
    assert torch.allclose(pruned.force_buffer[0, -1, :], wide[0, 0, :])
    assert torch.allclose(pruned.force_buffer[1, -1, :], wide[1, 0, :])


def test_frozen_pruned_scripted_round_trip(tmp_path):
    """The pruned scripted wrapper reproduces the frozen-latent semantics and
    contains strictly the components the deployment uses."""
    device = torch.device("cpu")
    torch.manual_seed(13)
    full = _make_spring_model(device, seed=13)
    full.eval()
    for p in full.parameters():
        p.requires_grad_(False)

    input_columns = ["measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_columns = ["tendon_bota_force_newton_data", "spring_coeff"]
    wrapped_full = ScaledModelWrapper(
        full,
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        frequency=200,
        history_size=4,
        stride=2,
        input_columns=input_columns,
        output_columns=output_columns,
    )
    wrapped_full.eval()
    full_path = os.path.join(tmp_path, "spring_full.pt")
    torch.jit.script(wrapped_full).save(full_path)
    loaded_full = torch.jit.load(full_path)

    frozen_latent = torch.randn(1, 1, 16)
    pruned = build_frozen_latent_model(loaded_full, frozen_latent)
    assert type(pruned).__name__ == "FrozenLatentForceModel"
    wrapped_pruned = ScaledModelWrapper(
        pruned,
        loaded_full.input_mean,
        loaded_full.input_std,
        loaded_full.output_mean,
        loaded_full.output_std,
        frequency=loaded_full.metadata["frequency"],
        history_size=loaded_full.metadata["history_size"],
        stride=loaded_full.metadata["stride"],
        input_columns=loaded_full.input_columns,
        output_columns=loaded_full.output_columns,
    )
    wrapped_pruned.eval()
    pruned_path = os.path.join(tmp_path, "spring_pruned.pt")
    torch.jit.script(wrapped_pruned).save(pruned_path)

    # The pruned checkpoint drops everything the deployment does not use.
    loaded_pruned = torch.jit.load(pruned_path)
    assert not hasattr(loaded_pruned.model, "model_transformer")
    assert not hasattr(loaded_pruned.model, "spring_buffer")
    assert not hasattr(loaded_pruned.model, "last_latent")
    assert not hasattr(loaded_pruned.model, "latent_anchor")
    assert not hasattr(loaded_pruned.model, "anchor_valid")
    assert not hasattr(loaded_pruned.model, "spring_update_counter")
    assert not hasattr(loaded_pruned.model, "spring_stride")
    assert not hasattr(loaded_pruned.model, "velocity_threshold_lo")
    assert hasattr(loaded_pruned.model, "force_transformer")
    assert hasattr(loaded_pruned.model, "spring_coeff_head")
    assert hasattr(loaded_pruned.model, "force_buffer")
    assert hasattr(loaded_pruned.model, "frozen_latent")

    # Tick-for-tick equivalence with the manual reference.
    ticks = [_tick(0.0, 0.0), _tick(0.2, 1.0), _tick(0.4, -1.0)]
    loaded_pruned.reset()
    state = {"force_buffer": None}
    for i, x in enumerate(ticks):
        out = loaded_pruned(x)
        ref = _frozen_reference(loaded_pruned.model, x, frozen_latent, state)
        assert torch.allclose(out, ref, atol=1e-6), f"tick {i}"

    # The embedded latent survives reset and batch-size changes.
    loaded_pruned.reset()
    assert torch.allclose(loaded_pruned.model.frozen_latent, frozen_latent, atol=1e-6)
    wide = torch.cat([ticks[0], ticks[1]], dim=0)
    out_wide = loaded_pruned(wide)
    assert out_wide.shape == (2, 1, 2)

    # set_frozen_latent can retarget the latent on the loaded module.
    state = {"force_buffer": None}
    out_first = loaded_pruned(ticks[0])
    ref = _frozen_reference(loaded_pruned.model, ticks[0], frozen_latent, state)
    assert torch.allclose(out_first, ref, atol=1e-6)
    new_latent = torch.randn(1, 1, 16)
    loaded_pruned.model.set_frozen_latent(new_latent)
    loaded_pruned.reset()
    state = {"force_buffer": None}
    out_retargeted = loaded_pruned(ticks[0])
    ref_new = _frozen_reference(loaded_pruned.model, ticks[0], new_latent, state)
    assert torch.allclose(out_retargeted, ref_new, atol=1e-6)
    assert not torch.allclose(out_retargeted, out_first, atol=1e-5)


def test_build_frozen_latent_model_rejects_pruned_checkpoint(tmp_path):
    """The export helper requires the full spring transformer checkpoint."""
    device = torch.device("cpu")
    torch.manual_seed(14)
    full = _make_spring_model(device, seed=14)
    full.eval()
    for p in full.parameters():
        p.requires_grad_(False)
    pruned_path = os.path.join(tmp_path, "spring_pruned.pt")
    torch.jit.script(
        ScaledModelWrapper(
            FrozenLatentForceModel(
                force_transformer=full.force_transformer,
                spring_coeff_head=full.spring_coeff_head,
                latent_dim=16,
                input_size=2,
                force_history_size=4,
            ),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            frequency=200,
            history_size=4,
            stride=2,
            input_columns=["measured_position_rad_data", "measured_velocity_rad_per_sec_data"],
            output_columns=["tendon_bota_force_newton_data", "spring_coeff"],
        )
    ).save(pruned_path)

    loaded_pruned = torch.jit.load(pruned_path)
    try:
        build_frozen_latent_model(loaded_pruned, torch.randn(1, 1, 16))
    except ValueError as e:
        assert "already pruned" in str(e)
    else:
        raise AssertionError("Expected ValueError for an already-pruned checkpoint")


def test_compute_frozen_latent_from_test_mcap():
    """The exporter returns a normalized mean latent over the recording's windows."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    device = torch.device("cpu")
    model_transformer = TorchTransformerModel(
        input_size=2,
        output_size=4,
        num_layers=1,
        history_size=4,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=2 + 4,
        output_size=1,
        num_layers=1,
        history_size=1,
        num_heads=2,
        hidden_dim=8,
        device=device,
    )
    spring_coeff_head = SpringCoefficientHead(latent_dim=4, device=device)
    estimator = SpringTransformerModel(
        model_transformer=model_transformer,
        force_transformer=force_transformer,
        spring_coeff_head=spring_coeff_head,
        latent_dim=4,
    )
    wrapped = ScaledModelWrapper(
        estimator,
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        frequency=200,
        history_size=1,
        stride=20,
        input_columns=["measured_position_rad_data", "measured_velocity_rad_per_sec_data"],
        output_columns=["tendon_bota_force_newton_data", "spring_coeff"],
    )
    wrapped.eval()
    scripted = torch.jit.script(wrapped)

    frozen_latent, num_windows = compute_frozen_latent(scripted, [TEST_MCAP])

    assert frozen_latent.shape == (1, 1, 4)
    assert num_windows > 0

    # The transformer runs in eval mode: the result is deterministic.
    frozen_latent_second, num_windows_second = compute_frozen_latent(scripted, [TEST_MCAP])
    assert num_windows_second == num_windows
    assert torch.allclose(frozen_latent_second, frozen_latent, atol=1e-6)
