import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import Tensor


class ScaledModelWrapper(nn.Module):
    """
    A PyTorch wrapper that:
    1. Applies input normalization & output denormalization
    2. Supports freezing the model
    3. Can be exported as TorchScript via torch.jit.script (scaling is included in the exported model)
    """

    def __init__(
        self,
        model: nn.Module,
        input_mean: Tensor,
        input_std: Tensor,
        output_mean: Tensor,
        output_std: Tensor,
        frequency: int = 1,
        history_size: int = 1,
        stride: int = 1,
        input_columns: list[str] = [],
        output_columns: list[str] = [],
    ):
        super().__init__()
        self.model = model

        self.model_type = type(model).__name__

        # Register scaling as buffers (so they're saved in state_dict)
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)
        self.metadata: dict[str, int] = {
            "frequency": frequency,
            "history_size": history_size,
            "stride": stride,
        }
        if hasattr(model, "rnn") and model.rnn is not None:
            self.register_buffer("h0", torch.zeros(model.num_layers, 1, model.hidden_size))
        self.input_columns = input_columns
        self.output_columns = output_columns

    def reset(self):
        """Reset any internal state of the model (if applicable)"""
        if hasattr(self, "h0"):
            self.h0[:] = 0.0
        if hasattr(self.model, "reset") and callable(self.model.reset):
            self.model.reset()

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.input_mean) / self.input_std

        if hasattr(self, "h0"):
            x, self.h0 = self.model.forward(x, self.h0)
        else:
            x = self.model.forward(x)

        x = x * self.output_std + self.output_mean

        return x

    def freeze(self) -> None:
        """Freeze model weights and disable gradients."""
        self.eval()  # Disables dropout/BatchNorm training behavior
        self.model.eval()  # Disables dropout/BatchNorm training behavior
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze model weights and re-enable gradients on all parameters."""
        self.train()  # Re-enables BatchNorm running stats updates
        self.model.train()  # Re-enables BatchNorm running stats updates
        for param in self.parameters():
            param.requires_grad = True

    def script_and_save(self, save_path: str) -> torch.jit.ScriptModule:
        """
        Script the model (including scaling layers) with torch.jit.script and save as TorchScript.
        Args:
            save_path: Where to save the scripted model (.pt or .pth)
        """
        scripted_model = torch.jit.script(self)
        scripted_model.save(save_path)
        return scripted_model


class ModelSaver:
    _root_folder: str
    _folder: str
    _file_prefix: str
    _wrapped_model: ScaledModelWrapper

    def __init__(self, model: ScaledModelWrapper, folder: str):
        self._wrapped_model = model
        now = datetime.now()
        self._root_folder = folder
        prefix = now.strftime("%Y_%m_%d_%H_%M_%S")
        self._folder = os.path.join(self._root_folder, prefix + "/")
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        self._file_prefix = os.path.join(self._folder, prefix + "_")

    def save_model(self, suffix: str) -> None:
        """Save the model as a TorchScript file
        Args:
            model (ScaledModelWrapper): The model to save
            suffix (str): Suffix to append to the filename
        """
        if suffix.startswith("_"):
            suffix = suffix[1:]
        self._wrapped_model.freeze()
        save_path = self._file_prefix + suffix + ".pt"
        self._wrapped_model.script_and_save(save_path)
        self._wrapped_model.unfreeze()

    def save_latest(self, prefix: str) -> None:
        """Save the model as 'latest.pt' in the root folder"""
        self._wrapped_model.freeze()
        save_path = os.path.join(self._root_folder, f"{prefix}latest.pt")
        self._wrapped_model.script_and_save(save_path)
        self._wrapped_model.unfreeze()
