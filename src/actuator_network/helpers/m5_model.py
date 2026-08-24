"""M5 friction model for tendon-driven actuator force estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class M5FrictionModel(nn.Module):
    """M5 friction model.

    tau_friction = K_v * velocity
                 + K_c
                 + |K_m * tau_motor - K_e * tau_external|
                 + exp(-|velocity / V_s|^alpha)
                   * (K_cs + |K_ms * tau_motor - K_es * tau_external|)

    All ``K`` coefficients, ``V_s``, ``alpha`` and the motor gain ``P`` are
    constrained positive via softplus.
    """

    def __init__(
        self,
        motor_gain: float = 4.2,
        trainable_motor_gain: bool = False,
    ) -> None:
        super().__init__()
        self.K_v_log = nn.Parameter(self._inverse_softplus(torch.tensor(0.01)))
        self.K_c_log = nn.Parameter(self._inverse_softplus(torch.tensor(0.01)))
        self.K_m_log = nn.Parameter(self._inverse_softplus(torch.tensor(1.0)))
        self.K_e_log = nn.Parameter(self._inverse_softplus(torch.tensor(1.0)))
        self.V_s_log = nn.Parameter(torch.tensor(0.0))
        self.alpha_log = nn.Parameter(torch.tensor(0.0))
        self.K_cs_log = nn.Parameter(self._inverse_softplus(torch.tensor(0.01)))
        self.K_ms_log = nn.Parameter(self._inverse_softplus(torch.tensor(1.0)))
        self.K_es_log = nn.Parameter(self._inverse_softplus(torch.tensor(1.0)))

        # Motor gain P in tau_motor = P * delta_position. Kept positive via softplus.
        self.motor_gain_log = nn.Parameter(self._inverse_softplus(torch.tensor(motor_gain)))
        self.motor_gain_log.requires_grad = trainable_motor_gain

    @staticmethod
    def _inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Inverse of y = softplus(x), stabilized for small values."""
        return torch.log(torch.exp(x) - 1 + eps)

    def _motor_gain(self) -> torch.Tensor:
        """Return the constrained-positive motor gain."""
        eps = 1e-6
        return functional.softplus(self.motor_gain_log) + eps

    def motor_gain_value(self) -> torch.Tensor:
        """Public accessor for the current motor gain (useful for logging/saving)."""
        return self._motor_gain()

    def compute_tau_motor(self, delta_position: torch.Tensor) -> torch.Tensor:
        """Compute motor torque from position error using the learned/fixed gain."""
        return self._motor_gain() * delta_position

    def set_friction_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients on all friction parameters.

        The motor gain is not affected; use ``trainable_motor_gain`` at init or
        set ``self.motor_gain_log.requires_grad`` directly.
        """
        friction_params = [
            self.K_v_log,
            self.K_c_log,
            self.K_m_log,
            self.K_e_log,
            self.V_s_log,
            self.alpha_log,
            self.K_cs_log,
            self.K_ms_log,
            self.K_es_log,
        ]
        for param in friction_params:
            param.requires_grad = trainable

    def _constrained_params(self):
        """Return all constrained-positive friction parameters."""
        eps = 1e-6
        return (
            functional.softplus(self.K_v_log),
            functional.softplus(self.K_c_log),
            functional.softplus(self.K_m_log),
            functional.softplus(self.K_e_log),
            functional.softplus(self.V_s_log) + eps,
            functional.softplus(self.alpha_log) + eps,
            functional.softplus(self.K_cs_log),
            functional.softplus(self.K_ms_log),
            functional.softplus(self.K_es_log),
        )

    def forward(
        self,
        velocity: torch.Tensor,
        tau_motor: torch.Tensor,
        tau_external: torch.Tensor,
    ) -> torch.Tensor:
        k_v, k_c, k_m, k_e, v_s, alpha, k_cs, k_ms, k_es = self._constrained_params()

        static_part = k_v * velocity + k_c + torch.abs(k_m * tau_motor - k_e * tau_external)

        stribeck_envelope = torch.exp(-(torch.abs(velocity / v_s).clamp_min(1e-8) ** alpha))
        stribeck_part = stribeck_envelope * (k_cs + torch.abs(k_ms * tau_motor - k_es * tau_external))

        return static_part + stribeck_part

    def named_physical_parameters(self) -> dict[str, float]:
        """Return the current parameter values as a plain dict."""
        k_v, k_c, k_m, k_e, v_s, alpha, k_cs, k_ms, k_es = self._constrained_params()
        return {
            "K_v": float(k_v.item()),
            "K_c": float(k_c.item()),
            "K_m": float(k_m.item()),
            "K_e": float(k_e.item()),
            "V_s": float(v_s.item()),
            "alpha": float(alpha.item()),
            "K_cs": float(k_cs.item()),
            "K_ms": float(k_ms.item()),
            "K_es": float(k_es.item()),
            "motor_gain": float(self._motor_gain().item()),
        }

    def set_physical_parameters(self, params: dict[str, float]) -> None:
        """Set model parameters from a dict of physical (positive) values."""
        eps = 1e-6

        def _get(key: str, old_key: str) -> float:
            if key in params:
                return params[key]
            if old_key in params:
                return params[old_key]
            raise KeyError(f"Missing parameter '{key}'")

        with torch.no_grad():
            self.K_v_log.fill_(self._inverse_softplus(torch.tensor(_get("K_v", "Kv"))).item())
            self.K_c_log.fill_(self._inverse_softplus(torch.tensor(_get("K_c", "Kc"))).item())
            self.K_m_log.fill_(self._inverse_softplus(torch.tensor(_get("K_m", "Km"))).item())
            self.K_e_log.fill_(self._inverse_softplus(torch.tensor(_get("K_e", "Ke"))).item())
            self.K_cs_log.fill_(self._inverse_softplus(torch.tensor(_get("K_cs", "Kcs"))).item())
            self.K_ms_log.fill_(self._inverse_softplus(torch.tensor(_get("K_ms", "K_ms"))).item())
            self.K_es_log.fill_(self._inverse_softplus(torch.tensor(_get("K_es", "Kes"))).item())
            self.V_s_log.fill_(torch.log(torch.exp(torch.tensor(_get("V_s", "V_s") - eps)) - 1).item())
            self.alpha_log.fill_(torch.log(torch.exp(torch.tensor(_get("alpha", "alpha") - eps)) - 1).item())
            if "motor_gain" in params:
                self.motor_gain_log.fill_(self._inverse_softplus(torch.tensor(params["motor_gain"])).item())
