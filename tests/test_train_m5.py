"""Tests for the M5 friction envelope training pipeline."""

from unittest import mock

import numpy as np
import pandas as pd
import torch

from actuator_network.helpers.hyperparameters import M5FrictionConfig
from actuator_network.helpers.torch_model import M5EnvelopeFrictionModel
from actuator_network.train_m5 import FRICTION_COL, TAU_EXTERNAL_COL, TAU_MOTOR_COL, VELOCITY_COL, build_friction_samples, train_m5


def _make_df(velocity, tau_motor, tau_external, friction) -> pd.DataFrame:
    return pd.DataFrame({VELOCITY_COL: velocity, TAU_MOTOR_COL: tau_motor, TAU_EXTERNAL_COL: tau_external, FRICTION_COL: friction})


def test_model_parameters_roundtrip():
    init = {"K_v": 0.002, "K_c": 0.03, "v_s": 0.5, "alpha": 2.0}
    params = M5EnvelopeFrictionModel(init_params=init).physical_parameters()
    for name, value in init.items():
        assert abs(params[name] - value) < 1e-5


def test_build_friction_samples_masks_and_targets():
    velocity = np.array([0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5])
    friction = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8])
    df = _make_df(velocity, np.zeros(8), np.zeros(8), friction)

    samples = build_friction_samples([df], velocity_threshold=0.01, breakaway_min_static_samples=3, device="cpu")

    assert samples["moving"].tolist() == [False, False, False, True, True, False, False, True]
    # Index 2 ends a 3-sample static run; index 6 ends a run that is too short.
    assert samples["breakaway"].tolist() == [False, False, True, False, False, False, False, False]
    assert samples["static"].tolist() == [True, True, False, False, False, True, True, False]
    # Moving targets are sign-corrected (friction opposes velocity), static targets are |tau_f|.
    assert torch.allclose(samples["target"], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))


def test_breakaway_does_not_cross_dataframes():
    first = _make_df(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5))
    second = _make_df(np.full(5, 0.5), np.zeros(5), np.zeros(5), np.zeros(5))

    samples = build_friction_samples([first, second], velocity_threshold=0.01, breakaway_min_static_samples=1, device="cpu")

    assert not samples["breakaway"].any()


def test_build_friction_samples_dead_band_and_breakaway_window():
    # Static run, one dead-band sample, a 1-sample moving blip, static, dead band, then a real motion.
    velocity = np.array([0.0, 0.0, 0.0, 0.04, 0.5, 0.0, 0.0, 0.0, 0.04, 0.5, 0.5, 0.5])
    friction = np.array([0.1, 0.9, 0.2, 0.0, 0.0, 0.3, 0.7, -0.4, 0.0, -0.1, -0.1, -0.1])
    df = _make_df(velocity, np.zeros(12), np.zeros(12), friction)

    samples = build_friction_samples(
        [df],
        velocity_threshold=0.06,
        breakaway_min_static_samples=3,
        device="cpu",
        static_velocity_threshold=0.03,
        min_moving_samples=2,
        breakaway_window=2,
    )

    # Dead-band samples (3, 8) and the too-short moving blip (4) are dropped.
    assert samples["moving"].tolist() == [False] * 6 + [True] * 3
    # Only index 7 is a breakaway: the blip does not count as motion.
    assert samples["breakaway"].tolist() == [False] * 5 + [True] + [False] * 3
    # Breakaway target is max |tau_f| over the last two static samples.
    assert torch.allclose(samples["target"], torch.tensor([0.1, 0.9, 0.2, 0.3, 0.7, 0.7, 0.1, 0.1, 0.1]))


def test_model_velocity_deadzone():
    model = M5EnvelopeFrictionModel(velocity_deadzone=0.03)
    zeros = torch.zeros(3)
    with torch.no_grad():
        inside = model(torch.tensor([0.0, 0.02, -0.03]), zeros, zeros)
        outside = model(torch.tensor([0.05, -0.05, 0.02]), zeros, zeros)
    assert torch.allclose(inside, inside[0].expand(3))
    assert torch.allclose(outside[0], outside[1]) and outside[0] < inside[0]
    # The dead zone is saved with the model.
    loaded = M5EnvelopeFrictionModel()
    loaded.load_state_dict(model.state_dict())
    assert torch.isclose(loaded.velocity_deadzone, torch.tensor(0.03))


def test_train_m5_recovers_envelope(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    true_model = M5EnvelopeFrictionModel(
        init_params={"K_v": 0.005, "K_c": 0.02, "K_m": 0.05, "K_e": 0.05, "v_s": 0.2, "alpha": 1.5, "K_cs": 0.03, "K_ms": 0.1, "K_es": 0.1}
    )

    n = 4000
    velocity = rng.uniform(-2.0, 2.0, n)
    velocity[rng.random(n) < 0.3] = 0.0
    tau_motor = rng.uniform(-0.2, 0.2, n)
    tau_external = rng.uniform(-0.2, 0.2, n)
    with torch.no_grad():
        envelope = true_model(*(torch.tensor(a, dtype=torch.float32) for a in (velocity, tau_motor, tau_external))).numpy()
    # Moving friction opposes velocity; static friction sits somewhere inside the envelope.
    friction = np.where(velocity != 0.0, -np.sign(velocity) * envelope, rng.uniform(-1.0, 1.0, n) * envelope)
    df = _make_df(velocity, tau_motor, tau_external, friction)

    samples = build_friction_samples([df], velocity_threshold=0.01, breakaway_min_static_samples=1, device="cpu")
    # The synthetic data has no velocity dead zone.
    config = M5FrictionConfig(num_epochs=3000, learning_rate=0.02, patience=3000, static_velocity_threshold=0.0, velocity_threshold=0.01)

    with mock.patch("actuator_network.train_m5.wandb"), mock.patch("actuator_network.train_m5.OUTPUT_DIR", str(tmp_path)):
        model = train_m5(config, samples, samples, device="cpu")

    with torch.no_grad():
        moving = samples["moving"]
        prediction = model(samples["velocity"], samples["tau_motor"], samples["tau_external"])
        rmse = torch.sqrt(torch.mean((prediction[moving] - samples["target"][moving]) ** 2)).item()
    assert rmse < 5e-3
    assert (tmp_path / "m5_params.json").exists()
