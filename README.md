# actuator_network

A PyTorch package for estimating actuator tendon load from ROS2 MCAP bag data.

## What it does

`actuator_network` reads bagged hardware experiments (motor positions, velocities, and load-cell / weight-sensor data), resamples and processes the signals, and trains small neural networks to estimate the tendon force in Newtons. Trained models are exported as TorchScript for deployment.

Three model architectures are supported:

- **MLP** — feed-forward network over a fixed history window
- **RNN** — recurrent network that maintains hidden state
- **Transformer** — causal transformer encoder over a history window

## Quick start

The recommended environment is Docker:

```bash
# CPU
docker compose up -d dev_cpu
docker exec -it actuator_network_cpu bash

# GPU
docker compose up -d dev_gpu
docker exec -it actuator_network_gpu bash
```

Inside the container, the repo is mounted at `/workspace`.

### Train a model

```bash
cd /workspace/src/actuator_network
python train_transformer.py   # or train_mlp.py / train_rnn.py
```

Training reads hardcoded MCAP paths under `data/training_data/`, resamples them to 80 Hz, derives velocity/acceleration/load columns, builds history windows, normalizes, and trains with Weights & Biases logging. Checkpoints are saved to `data/output_data/`.

### Run inference

```bash
python test.py
```

`test.py` loads `data/output_data/final_latest.pt`, inspects its embedded metadata, and writes a `_predicted.mcap` with the new `*_predicted` columns.

### Generate plots

```bash
cd /workspace/src/actuator_network/plots
python plot_rmse.py
```

Plot scripts are tailored to specific paper figures and assume the corresponding prediction MCAPs already exist.

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

## Configuration notes

- Training scripts contain hardcoded MCAP paths, model hyperparameters, and input/output column lists. Treat them as experiment notebooks rather than a generic CLI.
- `docker-compose.yml` currently contains an inline W&B API key. Move it to an `.env` file and rotate the key before sharing the project.

## License

MIT License — see [LICENSE](./LICENSE).
