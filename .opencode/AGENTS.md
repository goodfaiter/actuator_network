# Agent Notes: actuator_network

This file documents the `actuator_network` project so that agents can work on it effectively without re-discovering the architecture each time.

## Project purpose

`actuator_network` is a PyTorch-based Python package that processes ROS2 MCAP bag files and trains neural networks to estimate or predict actuator tendon load (force in Newtons). It is intended for hardware experiments where load cells / weight sensors and motor state (desired position, measured position, measured velocity) are logged as ROS2 topics.

Supported model families:

- MLP (`TorchMlpModel`)
- RNN (`TorchRNNModel`)
- Transformer (`TorchTransformerModel`)
- M5 + Transformer physics-coupled model (`M5TransformerPhysicsModel`)

The trained model is wrapped in `ScaledModelWrapper`, which includes input/output normalization, and exported as TorchScript for deployment.

## Repository layout

```
/workspace
├── README.md                           # Human-facing project overview
├── pyproject.toml                      # Package metadata + uv config
├── uv.lock                             # Locked dependency tree
├── .env.example                        # WANDB_API_KEY template
├── entrypoint.sh                       # Docker entrypoint: uv sync + exec
├── docker-compose.yml                  # dev service with GPU reservation
├── Dockerfile                          # GPU container (CUDA 12.8, ROS2 Humble, uv, opencode)
├── src/actuator_network/               # Main package
│   ├── __init__.py
│   ├── train_mlp.py                    # Entry point: train MLP
│   ├── train_rnn.py                    # Entry point: train RNN
│   ├── train_transformer.py            # Entry point: train Transformer
│   ├── train_m5.py                     # Entry point: fit M5 friction model
│   ├── train_m5_transformer.py         # Entry point: train M5 + Transformer physics-coupled model
│   ├── test_transformer.py             # Entry point: run Transformer inference on test MCAPs
│   ├── test_m5.py                      # Entry point: run M5 inference on test MCAPs
│   ├── test_m5_transformer.py          # Entry point: run M5 + Transformer inference on test MCAPs
│   ├── helpers/
│   │   ├── mcap_to_pandas.py           # Read ROS2 MCAP → pandas DataFrame
│   │   ├── pandas_processing.py        # Resample, derive load, filter, derivative
│   │   ├── pandas_to_torch.py          # Build history windows / sequences, normalize
│   │   ├── pandas_to_mcap.py           # Write DataFrame columns back to MCAP
│   │   ├── torch_model.py              # MLP, RNN, Transformer, M5+Transformer definitions
│   │   ├── trainer.py                  # Custom training loop with W&B logging
│   │   └── wrapper.py                  # ScaledModelWrapper + ModelSaver + TorchScript export
│   └── plots/                          # Matplotlib scripts for paper figures
│       ├── plot_contacts.py
│       ├── plot_contacts_sine_contact.py
│       ├── plot_ee_tracking.py
│       ├── plot_ramp.py
│       └── plot_rmse.py
├── data/                               # Data directory (gitignored)
│   ├── training_data/                  # Input MCAP files
│   └── output_data/                    # Saved models and predictions
└── .opencode/                          # opencode configuration
    └── AGENTS.md                       # This file
```

## Environment and dependencies

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. The lockfile targets the CUDA 12.8 build of PyTorch.

### Local uv workflow

```bash
# Sync dependencies and install the package in editable mode
uv sync
uv pip install -e . --link-mode=copy

# Run commands
uv run train-transformer
uv run predict
```

### Console scripts

`pyproject.toml` exposes these scripts:

- `train-mlp`
- `train-rnn`
- `train-transformer`
- `train-m5`
- `train-m5-transformer`
- `test-mlp`
- `test-rnn`
- `test-transformer`
- `test-m5`
- `test-m5-transformer`
- `predict`

Run them with `uv run <script>`.

### Docker workflow (optional)

A fully provisioned GPU environment is available via Docker:

```bash
docker compose up -d dev
docker exec -it actuator_network bash
```

The container entrypoint (`entrypoint.sh`) runs `uv sync` and `uv pip install -e .` automatically, then execs the container command.

### Weights & Biases

Training logs to W&B. The key is loaded from `.env` for Docker and can be exported locally:

```bash
cp .env.example .env
# edit .env
export WANDB_API_KEY=your_key_here
```

**Never commit the `.env` file or any API key.** It is already gitignored.

## Typical workflow

### 1. Prepare training data

Place ROS2 MCAP files under `data/training_data/<date>/`. The expected logged topics are:

- `/desired_position_rad` (`std_msgs/Float32`)
- `/measured_position_rad` (`std_msgs/Float32`)
- `/measured_velocity_rad_per_sec` (`std_msgs/Float32`)
- `/weight_kg` (`std_msgs/Float32`)
- `/imu/data_raw` (`sensor_msgs/Imu`) — currently parsed but not used in modeling

### 2. Train a model

```bash
uv run train-transformer
```

Each training script:

1. Reads every MCAP in its hardcoded list.
2. Resamples to the configured frequency (usually 80 or 200 Hz).
3. Computes derived columns (velocity, acceleration, dynamic force, load).
4. Writes a `_processed.mcap` next to each input file.
5. Builds history windows / sequences.
6. Normalizes inputs and outputs.
7. Trains with a 90/10 train/val split, MSE loss, Adam optimizer, and logs to Weights & Biases.
8. Saves the best, final, and periodic checkpoints as TorchScript `.pt` files in `data/output_data/`.

The `train_m5_transformer.py` script loads a pre-fit M5 friction model from `data/output_data/m5_friction_params.json` as an initial guess, then jointly trains the Transformer and (optionally) the M5 parameters so that the final output is `tau_external_calculated = tau_motor - tau_friction(tau_external_predicted)`. It uses an auxiliary loss on the Transformer's `tau_external` prediction and gradient clipping, and saves the final fitted M5 parameters to `data/output_data/m5_joint_friction_params.json`. Set `m5_trainable = False` in the script to keep M5 frozen as a fixed physics prior.

Key configuration knobs in the training scripts:

- `freq` / `data_freq` — target resampling frequency in Hz.
- `stride` — step size between history samples (e.g. `4` reduces 80 Hz data to 20 Hz inference).
- `num_hist` / `history_size` / `seq_length` — how many past samples the model sees.
- `prediction` — `False` means estimation at the current timestep; `True` would shift labels forward.
- `input_cols` / `output_cols` — which DataFrame columns are used.

### 3. Run inference

```bash
uv run test-mlp
uv run test-rnn
uv run test-transformer
uv run test-m5
uv run test-m5-transformer
```

Each `test-*.py` script loads the matching `best_<model>_latest.pt` TorchScript model (e.g., `best_transformer_latest.pt`), inspects its stored metadata (frequency, history size, stride, model type, input/output columns), builds the matching input tensor, runs the model, and writes a `<input>_<model>_predicted.mcap` with the new `*_predicted` columns.

| Model | Checkpoint loaded | MCAP suffix |
|---|---|---|
| MLP | `best_mlp_latest.pt` | `_mlp_predicted.mcap` |
| RNN | `best_rnn_latest.pt` | `_rnn_predicted.mcap` |
| Transformer | `best_transformer_latest.pt` | `_transformer_predicted.mcap` |
| M5 | `m5_friction_params.json` | `_m5_predicted.mcap` |
| M5 + Transformer | `best_m5_transformer_latest.pt` | `_m5_transformer_predicted.mcap` |

### 4. Generate plots

Plot scripts are self-contained Matplotlib notebooks that read prediction MCAPs from hardcoded paths and save PNGs to `src/actuator_network/plots/figures/`. Run individually, e.g.:

```bash
cd src/actuator_network/plots
uv run python plot_rmse.py
```

## Code conventions

- Use absolute paths under `/workspace` (or the repo root) in experiment scripts; data lives in `data/`.
- Always import from the package namespace: `from actuator_network.helpers...`.
- Keep model definitions in `helpers/torch_model.py`; do not add training logic there.
- Keep data I/O in `helpers/mcap_to_pandas.py` and `helpers/pandas_to_mcap.py`.
- Use `ScaledModelWrapper` as the deployment-facing model; it stores normalization statistics and model metadata as buffers so they are embedded in the TorchScript export.
- Prefer `torch.jit.script` over `trace` for the wrapper because it handles control flow and RNN state.
- Do not commit `.pt`, `.pth`, MCAP files, `__pycache__`, `.env`, `.venv`, or `wandb/` runs (they are already gitignored).
- Run `uv run ruff check src` and `uv run ruff format src` before finishing non-trivial changes.

## Known issues and gotchas

1. **Training scripts are hardcoded experiment notebooks.** Paths, model configs, and input/output columns are defined inside `train_mlp.py`, `train_rnn.py`, `train_transformer.py`, and `train_m5_transformer.py`. They work as `uv run train-*` entry points but are not a generic CLI yet.

2. **`process_inputs_time_series` zero-padding.** When the sequence start is before index `0`, the remaining entries are left as zeros. Make sure this behavior is intentional for your windowing strategy.

3. **RNN hidden state.** `ScaledModelWrapper` registers `h0` only when the wrapped model has an `rnn` attribute. For deployment, call `model.reset()` to clear state between sequences.

4. **Plot scripts are hardcoded to specific experimental files.** They will fail on a fresh checkout without the matching `data/` contents. They are intended for reproducing paper figures, not as a generic plotting CLI.

5. **Tests exist under `tests/`.** The ROS2 environment installs pytest plugins that conflict with plain `uv run pytest`, so disable plugin autoloading when running tests:
   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
   ```
   Agents adding non-trivial logic should add small sanity checks and run the command above.

## Useful commands

```bash
# Sync / install
uv sync
uv pip install -e . --link-mode=copy

# Train
uv run train-mlp
uv run train-rnn
uv run train-transformer
uv run train-m5
uv run train-m5-transformer

# Inference
uv run test-mlp
uv run test-rnn
uv run test-transformer
uv run test-m5
uv run test-m5-transformer

# Plots
cd src/actuator_network/plots
uv run python plot_rmse.py

# Lint / format
uv run ruff check src tests
uv run ruff format src tests

# Tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
```
