"""M5 friction model for tendon-driven actuator force estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class M5FrictionModel(nn.Module):
    """M5 friction model.

    tau_friction = Kv * velocity
                 + Kc
                 + |Km * tau_motor - Ke * tau_external|
                 + exp(-|velocity / V_s|^alpha)
                   * (Kcs + |K_ms * tau_motor - Kes * tau_external|)

    V_s and alpha are constrained positive via softplus. The motor gain P that
    maps delta_position to tau_motor can optionally be learned jointly.
    """

    def __init__(
        self,
        motor_gain: float = 4.2,
        trainable_motor_gain: bool = False,
    ) -> None:
        super().__init__()
        self.Kv = nn.Parameter(torch.tensor(0.01))
        self.Kc = nn.Parameter(torch.tensor(0.0))
        self.Km = nn.Parameter(torch.tensor(1.0))
        self.Ke = nn.Parameter(torch.tensor(1.0))
        self.V_s_log = nn.Parameter(torch.tensor(0.0))
        self.alpha_log = nn.Parameter(torch.tensor(0.0))
        self.Kcs = nn.Parameter(torch.tensor(0.0))
        self.K_ms = nn.Parameter(torch.tensor(1.0))
        self.Kes = nn.Parameter(torch.tensor(1.0))

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
            self.Kv,
            self.Kc,
            self.Km,
            self.Ke,
            self.V_s_log,
            self.alpha_log,
            self.Kcs,
            self.K_ms,
            self.Kes,
        ]
        for param in friction_params:
            param.requires_grad = trainable

    def _positive_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return constrained-positive parameters."""
        eps = 1e-6
        return functional.softplus(self.V_s_log) + eps, functional.softplus(self.alpha_log) + eps

    def forward(
        self,
        velocity: torch.Tensor,
        tau_motor: torch.Tensor,
        tau_external: torch.Tensor,
    ) -> torch.Tensor:
        v_s, alpha = self._positive_params()

        static_part = self.Kv * velocity + self.Kc + torch.abs(self.Km * tau_motor - self.Ke * tau_external)

        stribeck_envelope = torch.exp(-(torch.abs(velocity / v_s).clamp_min(1e-8) ** alpha))
        stribeck_part = stribeck_envelope * (self.Kcs + torch.abs(self.K_ms * tau_motor - self.Kes * tau_external))

        return static_part + stribeck_part

    def named_physical_parameters(self) -> dict[str, float]:
        """Return the current parameter values as a plain dict."""
        v_s, alpha = self._positive_params()
        return {
            "Kv": float(self.Kv.item()),
            "Kc": float(self.Kc.item()),
            "Km": float(self.Km.item()),
            "Ke": float(self.Ke.item()),
            "V_s": float(v_s.item()),
            "alpha": float(alpha.item()),
            "Kcs": float(self.Kcs.item()),
            "K_ms": float(self.K_ms.item()),
            "Kes": float(self.Kes.item()),
            "motor_gain": float(self._motor_gain().item()),
        }

    def set_physical_parameters(self, params: dict[str, float]) -> None:
        """Set model parameters from a dict of physical (unconstrained) values."""
        eps = 1e-6
        with torch.no_grad():
            self.Kv.fill_(params["Kv"])
            self.Kc.fill_(params["Kc"])
            self.Km.fill_(params["Km"])
            self.Ke.fill_(params["Ke"])
            self.V_s_log.fill_(torch.log(torch.exp(torch.tensor(params["V_s"] - eps)) - 1).item())
            self.alpha_log.fill_(torch.log(torch.exp(torch.tensor(params["alpha"] - eps)) - 1).item())
            self.Kcs.fill_(params["Kcs"])
            self.K_ms.fill_(params["K_ms"])
            self.Kes.fill_(params["Kes"])
            if "motor_gain" in params:
                self.motor_gain_log.fill_(self._inverse_softplus(torch.tensor(params["motor_gain"])).item())
