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

    V_s and alpha are constrained positive via softplus.
    """

    def __init__(self) -> None:
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

        static_part = (
            self.Kv * velocity
            + self.Kc
            + torch.abs(self.Km * tau_motor - self.Ke * tau_external)
        )

        stribeck_envelope = torch.exp(
            -torch.abs(velocity / v_s).clamp_min(1e-8) ** alpha
        )
        stribeck_part = stribeck_envelope * (
            self.Kcs + torch.abs(self.K_ms * tau_motor - self.Kes * tau_external)
        )

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
