import torch

import wandb
from actuator_network.helpers.wrapper import ModelSaver


def split_data(inputs, outputs, train_ratio=0.9):
    """Split inputs and outputs into training and validation sets
    Args:
        inputs (torch.Tensor): Input tensor of shape (num_samples, input_dim)
        outputs (torch.Tensor): Output tensor of shape (num_samples, output_dim)
        train_ratio (float): Ratio of data to use for training
    Returns:
        tuple: (train_inputs, train_outputs, val_inputs, val_outputs)
    """
    num_samples = inputs.shape[0]
    train_size = int(num_samples * train_ratio)

    indices = torch.randperm(num_samples)
    inputs = inputs[indices]
    outputs = outputs[indices]

    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    val_inputs = inputs[train_size:]
    val_outputs = outputs[train_size:]

    return train_inputs, train_outputs, val_inputs, val_outputs


def data_generator(inputs, outputs, batch_size):
    """Generate batches of data
    Args:
        inputs (torch.Tensor): Input tensor of shape (num_samples, input_dim)
        outputs (torch.Tensor): Output tensor of shape (num_samples, output_dim)
        batch_size (int): Size of each batch
    Yields:
        tuple: (batch_inputs, batch_outputs)
    """
    num_samples = inputs.shape[0]
    indices = torch.randperm(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield inputs[batch_indices], outputs[batch_indices]


def train(model, inputs, outputs, model_saver: ModelSaver = None):
    """Train the model with validation and model checkpointing

    Args:
        model: The PyTorch model to train
        inputs: Input tensor
        outputs: Output tensor
        model_saver: ModelSaver instance for saving
    """
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 1024
    train_ratio = 0.9

    wandb.init(project="actuator_network")
    wandb.config.update(
        {"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs, "train_ratio": train_ratio}
    )
    wandb.log({"Model": str(model)})
    # wandb.watch(model, log="all", log_freq=100)

    # Split data
    inputs_train, outputs_train, inputs_val, outputs_val = split_data(inputs, outputs, train_ratio=train_ratio)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track best validation loss for checkpointing
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        data_generator_train = data_generator(inputs_train, outputs_train, batch_size)

        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_outputs in data_generator_train:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(inputs_val)
            if isinstance(val_predictions, tuple):
                val_predictions = val_predictions[0]
            val_loss = criterion(val_predictions, outputs_val)

        # Log metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")

        wandb.log({"train_loss": avg_train_loss, "val_loss": val_loss.item(), "epoch": epoch + 1})

        # Save every 100 epochs
        if (epoch + 1) % 100 == 0:
            model_saver.save_model(f"_epoch_{epoch + 1}")

        # Check if this is the best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            model_saver.save_model("_best")
            model_saver.save_latest("best_")
            print(f"New best model! Val loss: {best_val_loss:.4f}")

    model_saver.save_model("_final")
    model_saver.save_latest("final_")
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
            model_saver.save_latest("best_")
            print(f"New best model! Val loss: {best_val_loss:.4f}")

    model_saver.save_model("_final")
    model_saver.save_latest("final_")
    wandb.finish()
