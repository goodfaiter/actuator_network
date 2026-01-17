import torch
import torch.nn as nn
from torch import Tensor


class ScaledModelWrapper(nn.Module):
    """
    A PyTorch wrapper that:
    1. Applies input normalization & output denormalization
    2. Supports freezing the model
    3. Can be JIT-traced (scaling is included in the exported model)
    """

    def __init__(
        self,
        model: nn.Module,
        input_mean: Tensor,
        input_std: Tensor,
        output_mean: Tensor,
        output_std: Tensor,
    ):
        super().__init__()
        self.model = model

        # Register scaling as buffers (so they're saved in state_dict)
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.input_mean) / self.input_std

        x = self.model(x)

        x = x * self.output_std + self.output_mean

        return x

    def freeze(self) -> None:
        """Freeze model weights and disable gradients."""
        self.eval()  # Disables dropout/BatchNorm training behavior
        self.model.eval()  # Disables dropout/BatchNorm training behavior
        for param in self.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze model weights."""
        self.train()  # Re-enables BatchNorm running stats updates
        self.model.train()  # Re-enables BatchNorm running stats updates
        for param in self.parameters():
            param.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True

    def trace_and_save(self, save_path: str) -> torch.jit.ScriptModule:
        """
        Trace the model (including scaling layers) and save as TorchScript.
        Args:
            example_input: A sample input tensor (for tracing)
            save_path: Where to save the traced model (.pt or .pth)
        """
        tracedmodel = torch.jit.script(self)
        tracedmodel.save(save_path)
        return tracedmodel