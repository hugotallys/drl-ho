# drl-ho

Hands-on experiments following *Deep Reinforcement Learning Hands-On* (M. Lapan), starting with chapter 4 (cross-entropy method on CartPole).

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
