# AgoraForge

AgoraForge is research code for evaluating and searching oversight mechanisms in
synthetic graph environments. It trains small graph-transformer policies with PPO,
either under a fixed mechanism or across a population of candidate mechanisms scored
by SHADE.

The repository contains two environments:

- `debate`: two agents reveal claims from a sampled factor graph to a bounded judge;
- `math`: agents prove, conjecture, and query over a sampled theorem graph.

This is an experimental research artifact, not a stable library API.

## Installation

AgoraForge requires Python 3.11 or newer. PyTorch runs on CPU for tests and small
examples; substantial training and search runs are intended for CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Weights & Biases logging is optional:

```bash
pip install -e '.[wandb]'
```

Without that extra, runs write TensorBoard logs locally.

## Quick smoke run

This runs one small CPU training epoch under vanilla debate:

```bash
agoraforge-train \
  --name=smoke-debate \
  --config=agoraforge/conf/runs/debate/default.py \
  --config.device=cpu \
  --config.training.online_epochs=1 \
  --config.training.online_buffer_size=8 \
  --config.training.online_batch_size=64
```

Outputs are written to `results/smoke-debate/`:

```text
checkpoints/model_final.pt   trained actor/critic and model configuration
resolved_config.json         complete run and environment configuration
config_overrides.json        command-line overrides
level_curve.json             tail-averaged primary metric by level
tensorboard/                 local scalar logs
```

Inspect the logs with:

```bash
tensorboard --logdir results/smoke-debate/tensorboard
```

## Mechanism search

The smoke search configurations exercise cold policy training, population evaluation,
common random numbers, and SHADE on CPU-sized problems:

```bash
agoraforge-search \
  --config=agoraforge/conf/runs/debate/protocol_search_smoke.py \
  --out=results/smoke-search
```

Search output includes the resumable strategy state, per-generation metrics and
candidates, the current incumbent mechanism, and each generation's best candidate.
Full search configurations are under `agoraforge/conf/runs/<environment>/`; they are
substantially more expensive and assume CUDA where specified.

## Trajectory visualization

A trained checkpoint can be replayed into a standalone HTML file. The run config must
describe the environment the checkpoint was trained in; learned mechanisms can be
supplied separately when applicable.

```bash
agoraforge-viz-debate \
  --config=agoraforge/conf/runs/debate/learned_fresh.py \
  --actor-checkpoint=results/<run>/checkpoints/model_final.pt \
  --out=results/debate-trajectory.html

agoraforge-viz-math \
  --config=agoraforge/conf/runs/math/default.py \
  --actor-checkpoint=results/<run>/checkpoints/model_final.pt \
  --out=results/math-trajectory.html
```

## Configuration and reproducibility

Runs are Python config files built with `ml_collections`. Override an existing field
with `--config.<path>=<value>`; unknown paths are rejected. The resolved run config and
all overrides are saved with every training run.

Population search uses common random numbers within a generation: candidates share
environment instances, policy initialization, and sampling draws. Environment and
policy seeds change between generations, and every candidate policy is trained from
scratch. SHADE parents are therefore reevaluated beside their trials on the current
generation's seeds.

`training.deterministic=True` enables PyTorch deterministic algorithms. Exact
bit-reproduction still depends on the same hardware and software stack.

## Repository layout

```text
agoraforge/
  conf/             run configs and shared defaults
  envs/             debate and math environments
  models/           graph-transformer actor/critic
  training/         resident and population PPO
  searching/        SHADE and multi-GPU worker dispatch
  visualization/    standalone trajectory renderers
  train.py          training entry point
  search.py         mechanism-search entry point
tests/               semantic, numerical, and integration checks
```

## Tests

The test suite includes independent correctness oracles for the math environment,
exact-inference checks for debate, batched-versus-solo parity, PPO integration, search
behavior, and visualization round trips.

```bash
pip install -e '.[dev]'
pytest -q
```

## License

AgoraForge is released under the [MIT License](LICENSE).
