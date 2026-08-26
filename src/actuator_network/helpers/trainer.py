from typing import Callable

import torch

import wandb
from actuator_network.helpers.wrapper import ModelSaver


def split_data(inputs, outputs, train_ratio=0.9):
    """Split inputs and outputs into training and validation sets
    Args:
        inputs (torch.Tensor or tuple[torch.Tensor, ...]): Input tensor(s). When a tuple,
            all tensors must share the same first dimension and are indexed with the same
            permutation.
        outputs (torch.Tensor): Output tensor of shape (num_samples, output_dim)
        train_ratio (float): Ratio of data to use for training
    Returns:
        tuple: (train_inputs, train_outputs, val_inputs, val_outputs)
    """
    if isinstance(inputs, tuple):
        num_samples = inputs[0].shape[0]
    else:
        num_samples = inputs.shape[0]
    train_size = int(num_samples * train_ratio)

    indices = torch.randperm(num_samples)
    if isinstance(inputs, tuple):
        inputs = tuple(inp[indices] for inp in inputs)
    else:
        inputs = inputs[indices]
    outputs = outputs[indices]

    train_inputs = inputs[:train_size] if not isinstance(inputs, tuple) else tuple(inp[:train_size] for inp in inputs)
    train_outputs = outputs[:train_size]
    val_inputs = inputs[train_size:] if not isinstance(inputs, tuple) else tuple(inp[train_size:] for inp in inputs)
    val_outputs = outputs[train_size:]

    return train_inputs, train_outputs, val_inputs, val_outputs


def data_generator(inputs, outputs, batch_size):
    """Generate batches of data
    Args:
        inputs (torch.Tensor or tuple[torch.Tensor, ...]): Input tensor(s). When a tuple,
            all tensors must share the same first dimension and are indexed with the same
            batch indices.
        outputs (torch.Tensor): Output tensor of shape (num_samples, output_dim)
        batch_size (int): Size of each batch
    Yields:
        tuple: (batch_inputs, batch_outputs)
    """
    if isinstance(inputs, tuple):
        num_samples = inputs[0].shape[0]
    else:
        num_samples = inputs.shape[0]
    indices = torch.randperm(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        if isinstance(inputs, tuple):
            batch_inputs = tuple(inp[batch_indices] for inp in inputs)
        else:
            batch_inputs = inputs[batch_indices]
        yield batch_inputs, outputs[batch_indices]


def train(
    model,
    inputs,
    outputs,
    model_saver: ModelSaver = None,
    latest_prefix: str = "",
    loss_fn=None,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 1024,
    train_ratio: float = 0.9,
    weight_decay: float = 0.0,
    scheduler_type: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.5,
    max_grad_norm: float | None = None,
    input_transform: Callable | None = None,
    accumulation_steps: int = 1,
):
    """Train the model with validation and model checkpointing

    Args:
        model: The PyTorch model to train
        inputs: Input tensor or tuple of tensors
        outputs: Output tensor
        model_saver: ModelSaver instance for saving
        latest_prefix: Optional prefix inserted before "best_"/"final_" for latest checkpoint names.
        loss_fn: Optional custom loss function. If None, MSE loss is used.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        train_ratio: Fraction of data to use for training.
        weight_decay: Adam weight decay (L2 penalty).
        scheduler_type: Learning-rate scheduler type. One of ``"none"``, ``"cosine"``, ``"step"``.
        scheduler_step_size: Period for StepLR, in epochs.
        scheduler_gamma: Multiplicative factor for StepLR.
        max_grad_norm: If provided, clip gradient norms to this value.
        input_transform: Optional transform applied to each training batch input
            before the forward pass. Should accept and return the same structure
            (tensor or tuple of tensors) as ``inputs``.
        accumulation_steps: Number of batches to accumulate gradients before an
            optimizer step. Effective batch size is ``batch_size * accumulation_steps``.
    """
    wandb.init(project="actuator_network")
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "num_epochs": num_epochs,
            "train_ratio": train_ratio,
            "weight_decay": weight_decay,
            "scheduler_type": scheduler_type,
        }
    )
    wandb.log({"Model": str(model)})
    # wandb.watch(model, log="all", log_freq=100)

    # Split data
    inputs_train, outputs_train, inputs_val, outputs_val = split_data(inputs, outputs, train_ratio=train_ratio)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning-rate scheduler
    scheduler = None
    scheduler_type_lower = scheduler_type.lower()
    if scheduler_type_lower == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type_lower == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Track best validation loss for checkpointing
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        data_generator_train = data_generator(inputs_train, outputs_train, batch_size)

        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        accumulated_batches = 0
        optimizer.zero_grad()

        for batch_inputs, batch_outputs in data_generator_train:
            if input_transform is not None:
                batch_inputs = input_transform(batch_inputs)

            if isinstance(batch_inputs, tuple):
                predictions = model(*batch_inputs)
            else:
                predictions = model(batch_inputs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if loss_fn is not None:
                loss = loss_fn(predictions, batch_outputs)
            else:
                loss = criterion(predictions, batch_outputs)
            loss = loss / accumulation_steps
            loss.backward()

            # Accumulate the unscaled loss for reporting.
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            accumulated_batches += 1

            if accumulated_batches == accumulation_steps:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accumulated_batches = 0

        # Step any leftover gradients at the end of the epoch.
        if accumulated_batches > 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation phase - run in batches to avoid OOM with long sequences.
        model.eval()
        val_loss_sum = 0.0
        val_num_batches = 0
        with torch.no_grad():
            for val_batch_inputs, val_batch_outputs in data_generator(inputs_val, outputs_val, batch_size):
                if isinstance(val_batch_inputs, tuple):
                    val_predictions = model(*val_batch_inputs)
                else:
                    val_predictions = model(val_batch_inputs)
                if isinstance(val_predictions, tuple):
                    val_predictions = val_predictions[0]
                if loss_fn is not None:
                    val_loss_sum += loss_fn(val_predictions, val_batch_outputs).item()
                else:
                    val_loss_sum += criterion(val_predictions, val_batch_outputs).item()
                val_num_batches += 1
        val_loss = val_loss_sum / max(val_num_batches, 1)

        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

        wandb.log({"train_loss": avg_train_loss, "val_loss": val_loss, "epoch": epoch + 1, "learning_rate": current_lr})

        # Save every 100 epochs
        if (epoch + 1) % 100 == 0:
            model_saver.save_model(f"_epoch_{epoch + 1}")

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_saver.save_model("_best")
            model_saver.save_latest(f"best_{latest_prefix}")
            print(f"New best model! Val loss: {best_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    model_saver.save_model("_final")
    model_saver.save_latest(f"final_{latest_prefix}")
    # Clean up wandb
    wandb.finish()


def train_stateful(
    model: torch.nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    model_saver: ModelSaver,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    chunk_batch_size: int = 4,
    max_grad_norm: float = 1.0,
    latest_prefix: str = "",
):
    """Train a recurrent model statefully with truncated backpropagation through time.

    Chunks are processed in consecutive batches. Gradients flow through all chunks
    within a batch; the hidden state is detached between batches to truncate gradients.

    Args:
        model: PyTorch RNN model whose `forward(chunk, h0)` returns `(pred, hn)`.
        train_inputs: Training input chunks of shape (num_chunks, seq_length, input_dim).
        train_targets: Training targets of shape (num_chunks, seq_length, output_dim).
        val_inputs: Validation input chunks.
        val_targets: Validation targets.
        model_saver: ModelSaver instance for checkpointing.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        chunk_batch_size: Number of consecutive chunks to process before a backward pass.
        max_grad_norm: Maximum gradient norm for clipping.
    """
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "chunk_batch_size": chunk_batch_size,
            "max_grad_norm": max_grad_norm,
        }
    )
    wandb.log({"Model": str(model)})

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    def _process_chunk(model, chunk, target, h0):
        if h0 is None:
            h0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=chunk.device)
        pred, hn = model.forward(chunk, h0)
        loss = criterion(pred, target)
        return loss, hn

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        h0 = None

        num_train_chunks = train_inputs.shape[0]
        for start in range(0, num_train_chunks, chunk_batch_size):
            end = min(start + chunk_batch_size, num_train_chunks)

            optimizer.zero_grad()
            batch_loss = 0.0

            for k in range(start, end):
                chunk = train_inputs[k : k + 1]
                target = train_targets[k : k + 1]

                loss, h0 = _process_chunk(model, chunk, target, h0)
                batch_loss += loss

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            # Truncated BPTT: detach hidden state between chunk batches
            h0 = h0.detach()

            epoch_loss += batch_loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation phase: also carry hidden state to match inference
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        h0 = None
        with torch.no_grad():
            num_val_chunks = val_inputs.shape[0]
            for start in range(0, num_val_chunks, chunk_batch_size):
                end = min(start + chunk_batch_size, num_val_chunks)

                batch_loss = 0.0
                for k in range(start, end):
                    chunk = val_inputs[k : k + 1]
                    target = val_targets[k : k + 1]

                    loss, h0 = _process_chunk(model, chunk, target, h0)
                    batch_loss += loss

                h0 = h0.detach()

                val_loss += batch_loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1})

        # Save every 100 epochs
        if (epoch + 1) % 100 == 0:
            model_saver.save_model(f"_epoch_{epoch + 1}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_saver.save_model("_best")
            model_saver.save_latest(f"best_{latest_prefix}")
            print(f"New best model! Val loss: {best_val_loss:.4f}")

    model_saver.save_model("_final")
    model_saver.save_latest(f"final_{latest_prefix}")
    wandb.finish()
