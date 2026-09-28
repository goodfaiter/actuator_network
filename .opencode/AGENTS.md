# Agent Notes: actuator_network

PyTorch package that processes ROS2 MCAP bag files and trains networks to estimate actuator tendon load (Newtons). Uses uv; the lockfile targets the CUDA 12.8 build of PyTorch.

## Layout

```
src/actuator_network/
├── train_mlp.py / train_rnn.py / train_transformer.py / ...   # Train + inference entry points (thin wrappers, hardcoded MCAP lists)
├── helpers/
│   ├── mcap_to_pandas.py       # ROS2 MCAP → pandas
│   ├── pandas_processing.py    # Resample (extrapolate_dataframe) + derived features; dt derived from index
│   ├── pandas_to_torch.py      # Windowing / sequences / normalization
│   ├── pandas_to_mcap.py       # pandas → MCAP
│   ├── torch_model.py          # Model definitions
│   ├── data_pipeline.py        # Parallel MCAP loading + processed DataFrame cache
│   ├── rnn_pipeline.py         # Stateful (chunked) inference helpers
│   ├── hyperparameters.py      # Dataclass configs, sweep-overridable via wandb.config
│   ├── trainer.py              # Training loops with W&B logging
│   └── wrapper.py              # ScaledModelWrapper + ModelSaver (TorchScript export)
├── plots/                      # Matplotlib figure scripts (hardcoded paths)
wandb_sweep/                    # W&B sweep YAML (train-spring-transformer is the sweep target)
tests/                          # pytest suite
```

## Commands

```bash
uv sync && uv pip install -e . --link-mode=copy

uv run train-mlp          # also: train-rnn, train-transformer,
                          # train-transformer-autoregressive, train-spring-transformer
uv run test-mlp           # also: test-rnn, test-transformer,
                          # test-transformer-autoregressive, test-spring-transformer,
                          # test-spring-transformer-frozen
uv run export-frozen-latent

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
uv run ruff check src tests && uv run ruff format src tests
```

## Conventions and gotchas

- Model definitions live in `helpers/torch_model.py`; training logic in the entry points; data I/O in `mcap_to_pandas.py` / `pandas_to_mcap.py`.
- Call `extrapolate_dataframe` before `process_dataframe` — the derivative timestep is derived from the index spacing.
- Deploy via `ScaledModelWrapper` (input/output normalization + a `metadata` dict with `frequency`/`history_size`/`stride`, embedded in the TorchScript export). Save with `ModelSaver` (`torch.jit.script`).
- `process_inputs_time_series` drops incomplete windows (no padding).
- `ScaledModelWrapper` only supports models whose `forward` takes a single `x` tensor (plus `h0` for RNNs); call `model.reset()` between sequences for RNNs.
- Expected logged topics: `/desired_position_rad`, `/measured_position_rad`, `/measured_velocity_rad_per_sec`, `/weight_kg`, `/bota/wrench_N_and_Nm`, `/imu/data_raw` (parsed but unused).
- Training logs to W&B; the key comes from `.env` (Docker) or `export WANDB_API_KEY=...`. Never commit `.env`, API keys, `.pt`, or MCAP files (all gitignored).
