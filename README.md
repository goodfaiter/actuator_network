# actuator_network

A PyTorch package for estimating actuator tendon load from ROS2 MCAP bag data.

## What it does

`actuator_network` reads bagged hardware experiments (motor positions, velocities, and load-cell / weight-sensor data), resamples and processes the signals, and trains small neural networks to estimate the tendon force in Newtons. Trained models are exported as TorchScript for deployment.

Seven model families are supported:

- **MLP** — feed-forward network over a fixed history window
- **RNN** — GRU network that maintains hidden state (truncated BPTT training)
- **Transformer** — causal transformer encoder over a history window
- **Autoregressive Transformer** — adds the (teacher-forced) force itself as an input channel; closed loop at inference
- **Plain M5 physics model** — friction model fitted from data, no neural network
- **M5 + Transformer** — physics-coupled: a Transformer predicts external torque, M5 computes friction, and the final output is `tau_motor - tau_friction` (train-only: saves the fitted friction params JSON, no TorchScript export)
- **Estimated-Spring Transformer** — dual (spring + force) transformer pair for variable spring stiffness, trained via a W&B sweep

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
# Training
uv run train-mlp
uv run train-rnn
uv run train-transformer
uv run train-m5
uv run train-m5-transformer
uv run train-transformer-autoregressive
uv run train-estimated-spring-transformer

# Inference (test_* scripts read frequency/history/stride metadata baked into the exported model)
uv run test-mlp
uv run test-rnn
uv run test-transformer
uv run test-m5
uv run test-transformer-autoregressive
uv run test-estimated-spring-transformer
```

The training hyperparameters live in `helpers/hyperparameters.py` as dataclasses (overridable from a W&B sweep config). Each entry point in `src/actuator_network/` hardcodes only its experiment MCAP path list, so treat them as experiment entry points rather than a generic CLI.

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
├── train_mlp.py / train_rnn.py / train_transformer.py / ...   # Train + inference entry points
├── helpers/
│   ├── mcap_to_pandas.py    # MCAP → pandas
│   ├── pandas_processing.py # Resampling & feature derivation
│   ├── pandas_to_torch.py   # Windowing & normalization
│   ├── pandas_to_mcap.py    # pandas → MCAP
│   ├── torch_model.py       # Model definitions
│   ├── m5_model.py          # M5 friction physics model
│   ├── data_pipeline.py     # Parallel MCAP loading + processed DataFrame cache
│   ├── rnn_pipeline.py      # Stateful (chunked) inference helpers
│   ├── hyperparameters.py   # Dataclass configs, sweep-overridable via wandb.config
│   ├── trainer.py           # Training loop
│   └── wrapper.py           # Normalization wrapper + TorchScript export
└── plots/                   # Matplotlib figure scripts
wandb_sweep/                  # W&B sweep configuration YAML
tests/                        # pytest suite
```

## Development

```bash
# Run tests (disable plugin autoloading: the ROS2 env installs conflicting pytest plugins)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/

# Run linting / formatting
uv run ruff check src
uv run ruff format src
```

## License

MIT License — see [LICENSE](./LICENSE).
