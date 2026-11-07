# drl-ho

Hands-on experiments following *Deep Reinforcement Learning Hands-On* (M. Lapan), starting with chapter 4 (CartPole).

## Setup

```bash
conda env create -f environment.yml
conda activate drl-ho
```

## Run

```bash
python src/random_cartpole.py --episodes 100 --seed 0   # headless, fast
python src/random_cartpole.py --episodes 5 --render     # with a window
```
