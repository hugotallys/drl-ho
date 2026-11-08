# drl-ho

Hands-on experiments following *Deep Reinforcement Learning Hands-On* by Maxim Lapan. The repo is a personal workspace for reimplementing the book's examples with modern tooling (Gymnasium, current PyTorch, ruff).

## What's here

| Script | Book chapter | Description |
| --- | --- | --- |
| `src/cartpole_cross_entropy.py` | 4 – The Cross-Entropy Method | A small policy network learns CartPole by playing batches of episodes, keeping only the best ones (above a reward percentile), and training to imitate their actions. |

Design notes:

- Each script is self-contained, with hyperparameters in a `Config` dataclass exposed as CLI flags.
- Headless by default for fast runs; video recording is opt-in.
- Runs are seeded (PyTorch, NumPy and the environment) for reproducibility.
- Training metrics (loss, mean reward, reward bound) are logged to TensorBoard.

## Setup

```bash
conda env create -f environment.yml
conda activate drl-ho
```

## Run

```bash
python src/cartpole_cross_entropy.py                    # headless, until solved
python src/cartpole_cross_entropy.py --max-iters 5      # quick smoke run
python src/cartpole_cross_entropy.py --record-video     # also save mp4 clips to ./video
python src/cartpole_cross_entropy.py --help             # all hyperparameters
```

Watch training curves:

```bash
tensorboard --logdir runs
```

## Develop

```bash
ruff check . && ruff format .
```
