# Agent Notes: actuator_network

This file documents the `actuator_network` project so that agents can work on it effectively without re-discovering the architecture each time.

## Project purpose

`actuator_network` is a PyTorch-based Python package that processes ROS2 MCAP bag files and trains neural networks to estimate or predict actuator tendon load (force in Newtons). It is intended for hardware experiments where load cells / weight sensors and motor state (desired position, measured position, measured velocity) are logged as ROS2 topics.

Supported model families:

- MLP (`TorchMlpModel`)
- RNN (`TorchRNNModel`)
- Transformer (`TorchTransformerModel`)

The trained model is wrapped in `ScaledModelWrapper`, which includes input/output normalization, and exported as TorchScript for deployment.

## Repository layout

```
/workspace
├── README.md                           # Human-facing project overview
├── pyproject.toml                      # Package metadata (name, version, python>=3.10)
├── docker-compose.yml                  # dev_cpu / dev_gpu services
├── DockerfileCpu                       # CPU-only container (Ubuntu 22.04, ROS2 Humble)
├── DockerfileGpu                       # GPU container (CUDA 12.8, ROS2 Humble)
├── src/actuator_network/               # Main package
│   ├── __init__.py
│   ├── train_mlp.py                    # Entry point: train MLP
│   ├── train_rnn.py                    # Entry point: train RNN
│   ├── train_transformer.py            # Entry point: train Transformer
│   ├── test.py                         # Entry point: run inference on test MCAPs
│   ├── helpers/
│   │   ├── mcap_to_pandas.py           # Read ROS2 MCAP → pandas DataFrame
│   │   ├── pandas_processing.py        # Resample, derive load, filter, derivative
│   │   ├── pandas_to_torch.py          # Build history windows / sequences, normalize
│   │   ├── pandas_to_mcap.py           # Write DataFrame columns back to MCAP
│   │   ├── torch_model.py              # MLP, RNN, Transformer definitions
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

The project is meant to run inside Docker:

```bash
# CPU container
docker compose up -d dev_cpu
docker exec -it actuator_network_cpu bash

# GPU container
docker compose up -d dev_gpu
docker exec -it actuator_network_gpu bash
```

Both containers install:

- ROS2 Humble (base)
- PyTorch (CPU or CUDA 12.8)
- Python packages: `numpy rosbags pybind11 pandas scikit-learn matplotlib tqdm roma PyQt6 pyarrow mcap mcap-ros2-support wandb`

Inside the container the workspace is mounted at `/workspace`.

### Environment variables

`docker-compose.yml` currently passes `WANDB_API_KEY` inline. **Do not commit new secrets to version control.** Prefer an `.env` file and `${WANDB_API_KEY}` interpolation in `docker-compose.yml`, and rotate any key that was previously committed.

## Typical workflow

### 1. Prepare training data

Place ROS2 MCAP files under `data/training_data/<date>/`. The expected logged topics are:

- `/desired_position_rad` (`std_msgs/Float32`)
- `/measured_position_rad` (`std_msgs/Float32`)
- `/measured_velocity_rad_per_sec` (`std_msgs/Float32`)
- `/weight_kg` (`std_msgs/Float32`)
- `/imu/data_raw` (`sensor_msgs/Imu`) — currently parsed but not used in modeling

### 2. Train a model

Run one of the training entry points from inside the container:

```bash
cd /workspace/src/actuator_network
python train_transformer.py
```

Each training script:

1. Reads every MCAP in its hardcoded list.
2. Resamples to the configured frequency (usually 80 Hz).
3. Computes derived columns (velocity, acceleration, dynamic force, load).
4. Writes a `_processed.mcap` next to each input file.
5. Builds history windows / sequences.
6. Normalizes inputs and outputs.
7. Trains with a 90/10 train/val split, MSE loss, Adam optimizer, and logs to Weights & Biases.
8. Saves the best, final, and periodic checkpoints as TorchScript `.pt` files in `data/output_data/`.

Key configuration knobs in the training scripts:

- `freq` / `data_freq` — target resampling frequency in Hz.
- `stride` — step size between history samples (e.g. `4` reduces 80 Hz data to 20 Hz inference).
- `num_hist` / `history_size` / `seq_length` — how many past samples the model sees.
- `prediction` — `False` means estimation at the current timestep; `True` would shift labels forward.
- `input_cols` / `output_cols` — which DataFrame columns are used.

### 3. Run inference

```bash
cd /workspace/src/actuator_network
python test.py
```

`test.py` loads `data/output_data/final_latest.pt` (TorchScript), inspects its stored metadata (frequency, history size, stride, model type, input/output columns), builds the matching input tensor, runs the model, and writes a `_predicted.mcap` with the new `*_predicted` columns.

### 4. Generate plots

Plot scripts are self-contained Matplotlib notebooks that read prediction MCAPs from hardcoded paths and save PNGs to `src/actuator_network/plots/figures/`. Run individually, e.g.:

```bash
cd /workspace/src/actuator_network/plots
python plot_rmse.py
```

## Code conventions

- Use absolute paths under `/workspace` in experiment scripts; data lives in `data/`.
- Keep model definitions in `helpers/torch_model.py`; do not add training logic there.
- Keep data I/O in `helpers/mcap_to_pandas.py` and `helpers/pandas_to_mcap.py`.
- Use `ScaledModelWrapper` as the deployment-facing model; it stores normalization statistics and model metadata as buffers so they are embedded in the TorchScript export.
- Prefer `torch.jit.script` over `trace` for the wrapper because it handles control flow and RNN state.
- Do not commit `.pt`, `.pth`, MCAP files, `__pycache__`, or `wandb/` runs (they are already gitignored).

## Known issues and gotchas

1. **Function signature mismatch.** `train_mlp.py` and `train_rnn.py` call `process_dataframe(data_df_extrapolated, spring_constant=spring_constant)`, but `helpers/pandas_processing.py::process_dataframe` only accepts `df`. This will raise a `TypeError` until the signature is reconciled.

2. **Type mismatch in `train_transformer.py`.** Its `mcap_file_paths` is a plain list of strings, while the other training scripts use a list of `(path, spring_constant)` tuples. The loop in `train_transformer.py` already iterates directly over strings, so it works, but the three training scripts are inconsistent.

3. **Hardcoded W&B API key.** `docker-compose.yml` contains an inline `WANDB_API_KEY`. Move it to an `.env` file and rotate the key.

4. **No tests or CI.** There is no test suite, linting, or formatting configuration yet. Agents adding non-trivial logic should add small sanity checks (e.g. shape tests for `process_inputs_time_series`).

5. **Plot scripts are hardcoded to specific experimental files.** They will fail on a fresh checkout without the matching `data/` contents. They are intended for reproducing paper figures, not as a generic plotting CLI.

6. **`process_inputs_time_series` zero-padding.** When the sequence start is before index `0`, the remaining entries are left as zeros. Make sure this behavior is intentional for your windowing strategy.

7. **RNN hidden state.** `ScaledModelWrapper` registers `h0` only when the wrapped model has an `rnn` attribute. For deployment, call `model.reset()` to clear state between sequences.

## Useful commands

```bash
# Inside the container
cd /workspace/src/actuator_network

# Train
python train_mlp.py
python train_rnn.py
python train_transformer.py

# Inference
python test.py

# Plots
cd /workspace/src/actuator_network/plots
python plot_rmse.py
```
