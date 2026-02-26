import torch
import wandb
from helpers.wrapper import ScaledModelWrapper, ModelSaver


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
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 64
    train_ratio = 0.9

    wandb.init(project="actuator_network")
    wandb.config.update({"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs, "train_ratio": train_ratio})
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
            val_loss = criterion(val_predictions, outputs_val)

        # Log metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], " f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")

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
