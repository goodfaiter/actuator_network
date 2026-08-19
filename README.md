# actuator_network

A PyTorch package for estimating actuator tendon load from ROS2 MCAP bag data.

## What it does

`actuator_network` reads bagged hardware experiments (motor positions, velocities, and load-cell / weight-sensor data), resamples and processes the signals, and trains small neural networks to estimate the tendon force in Newtons. Trained models are exported as TorchScript for deployment.

Three model architectures are supported:

- **MLP** — feed-forward network over a fixed history window
- **RNN** — recurrent network that maintains hidden state
- **Transformer** — causal transformer encoder over a history window

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the lockfile and install the package in editable mode
uv sync
uv pip install -e . --link-mode=copy

# 3. Run a training script
uv run train-transformer
```

The lockfile now targets the CUDA 12.8 build of PyTorch.

### Available commands

After syncing, the following console scripts are available via `uv run`:

```bash
uv run train-mlp
uv run train-rnn
uv run train-transformer
uv run predict
```

The training scripts still contain hardcoded MCAP paths and hyperparameters, so treat them as experiment entry points rather than a generic CLI.

### Weights & Biases

Training logs to W&B. Copy the example environment file and add your key:

```bash
cp .env.example .env
# edit .env with your WANDB_API_KEY
```

`docker compose` will pick it up automatically. For local `uv run`, export it:

```bash
export WANDB_API_KEY=your_key_here
```

### Docker (optional)

A Docker setup is provided for a fully provisioned GPU environment (Ubuntu 22.04 + CUDA 12.8 + cuDNN + ROS2 Humble + uv + opencode).

```bash
docker compose up -d dev
docker exec -it actuator_network bash
```

The container entrypoint runs `uv sync` and `uv pip install -e .` automatically, so the package is ready to use.

## Repository layout

```
src/actuator_network/
├── train_mlp.py / train_rnn.py / train_transformer.py   # Training entry points
├── test.py                                              # Inference entry point
├── helpers/
│   ├── mcap_to_pandas.py    # MCAP → pandas
│   ├── pandas_processing.py # Resampling & feature derivation
│   ├── pandas_to_torch.py   # Windowing & normalization
│   ├── pandas_to_mcap.py    # pandas → MCAP
│   ├── torch_model.py       # Model definitions
│   ├── trainer.py           # Training loop
│   └── wrapper.py           # Normalization wrapper + TorchScript export
└── plots/                   # Matplotlib figure scripts
```

## Development

```bash
# Run linting / formatting
uv run ruff check src
uv run ruff format src
```

## License

MIT License — see [LICENSE](./LICENSE).
