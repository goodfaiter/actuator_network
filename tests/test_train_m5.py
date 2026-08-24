import torch

from actuator_network.helpers.m5_model import M5FrictionModel


def _set_known_params(model: M5FrictionModel) -> None:
    """Set a deterministic, physically-positive parameter set."""
    model.set_physical_parameters(
        {
            "K_v": 0.1,
            "K_c": 0.2,
            "K_m": 0.3,
            "K_e": 0.4,
            "V_s": 0.693147 + 1e-6,  # matches softplus(0.0) + eps
            "alpha": 0.693147 + 1e-6,
            "K_cs": 0.5,
            "K_ms": 0.6,
            "K_es": 0.7,
        }
    )
    # V_s_log and alpha_log were fixed at 0.0 in the original manual formula.
    with torch.no_grad():
        model.V_s_log.fill_(0.0)
        model.alpha_log.fill_(0.0)


def test_m5_forward_matches_formula():
    """The model output should match a manual implementation of the M5 formula."""
    model = M5FrictionModel()
    _set_known_params(model)

    velocity = torch.tensor([0.0, 1.0, -2.0])
    tau_motor = torch.tensor([0.5, -0.5, 1.0])
    tau_external = torch.tensor([0.0, 1.0, -0.5])

    prediction = model(velocity, tau_motor, tau_external)

    # Manual computation
    eps = 1e-6
    v_s = torch.nn.functional.softplus(torch.tensor(0.0)) + eps
    alpha = torch.nn.functional.softplus(torch.tensor(0.0)) + eps
    static = 0.1 * velocity + 0.2 + torch.abs(0.3 * tau_motor - 0.4 * tau_external)
    stribeck = torch.exp(-(torch.abs(velocity / v_s).clamp_min(1e-8) ** alpha))
    expected = static + stribeck * (0.5 + torch.abs(0.6 * tau_motor - 0.7 * tau_external))

    assert torch.allclose(prediction, expected, atol=1e-5)


def test_m5_positive_parameters_are_enforced():
    """All K coefficients, V_s and alpha returned by the model must be positive."""
    model = M5FrictionModel()

    # Set unconstrained log parameters to negative values
    with torch.no_grad():
        model.K_v_log.fill_(-5.0)
        model.K_c_log.fill_(-10.0)
        model.K_m_log.fill_(-3.0)
        model.K_e_log.fill_(-4.0)
        model.V_s_log.fill_(-5.0)
        model.alpha_log.fill_(-10.0)
        model.K_cs_log.fill_(-2.0)
        model.K_ms_log.fill_(-1.0)
        model.K_es_log.fill_(-2.0)

    params = model.named_physical_parameters()

    assert params["K_v"] > 0.0
    assert params["K_c"] > 0.0
    assert params["K_m"] > 0.0
    assert params["K_e"] > 0.0
    assert params["V_s"] > 0.0
    assert params["alpha"] > 0.0
    assert params["K_cs"] > 0.0
    assert params["K_ms"] > 0.0
    assert params["K_es"] > 0.0


def test_m5_fitting_reduces_loss():
    """A few Adam steps should reduce MSE on synthetic data."""
    torch.manual_seed(0)
    model = M5FrictionModel()

    velocity = torch.randn(100)
    tau_motor = torch.randn(100)
    tau_external = torch.randn(100)

    # Synthetic target from a known parameter set
    with torch.no_grad():
        target_model = M5FrictionModel()
        _set_known_params(target_model)
        target = target_model(velocity, tau_motor, tau_external)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial_loss = torch.nn.functional.mse_loss(model(velocity, tau_motor, tau_external), target).item()

    for _ in range(200):
        optimizer.zero_grad()
        pred = model(velocity, tau_motor, tau_external)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(model(velocity, tau_motor, tau_external), target).item()
    assert final_loss < initial_loss
