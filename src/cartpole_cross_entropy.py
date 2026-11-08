"""Cross-entropy method on CartPole (Lapan, Deep RL Hands-On, chapter 4).

Each iteration plays a batch of episodes with the current policy, keeps only
the "elite" episodes (return at or above a percentile of the batch), and trains
the network to imitate the elite actions with a cross-entropy loss.

Runs headless by default for speed; pass --record-video to save mp4 clips.
Metrics are written to TensorBoard under ./runs.
"""

import argparse
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class Config:
    """All hyperparameters and run options in one place."""

    env_id: str = "CartPole-v1"
    hidden_size: int = 128
    batch_size: int = 16
    percentile: float = 70
    lr: float = 0.01
    seed: int = 123456
    max_iters: int | None = None
    record_video: bool = False
    video_dir: str = "video"


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-id", default=defaults.env_id)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--percentile", type=float, default=defaults.percentile)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=defaults.max_iters,
        help="stop after this many iterations (default: until solved)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="save mp4 clips (slower; uses the default RecordVideo schedule)",
    )
    parser.add_argument("--video-dir", default=defaults.video_dir)
    return Config(**vars(parser.parse_args()))


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: list[EpisodeStep]


def make_env(cfg: Config) -> gym.Env:
    """Build the environment; rendering is only enabled when recording video."""
    env = gym.make(cfg.env_id, render_mode="rgb_array" if cfg.record_video else None)
    if cfg.record_video:
        env = gym.wrappers.RecordVideo(env, video_folder=cfg.video_dir)
    return env


def iterate_batches(
    env: gym.Env, net: Net, batch_size: int, seed: int
) -> Iterator[list[Episode]]:
    """Endlessly play episodes with `net` and yield them in batches."""
    batch: list[Episode] = []
    episode_reward = 0.0
    episode_steps: list[EpisodeStep] = []
    # Seed the env only on the first reset; reseeding every reset would make
    # all episodes identical.
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    while True:
        with torch.no_grad():
            obs_v = torch.as_tensor(obs, dtype=torch.float32)
            act_probs = torch.softmax(net(obs_v), dim=-1).numpy()

        action = int(rng.choice(len(act_probs), p=act_probs))
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        episode_reward += float(reward)
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done or is_trunc:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def filter_batch(
    batch: list[Episode], percentile: float
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Keep episodes at/above the reward percentile; flatten them into tensors.

    Returns (observations, actions, reward_bound, reward_mean).
    """
    rewards = [episode.reward for episode in batch]
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs: list[np.ndarray] = []
    train_act: list[int] = []

    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(step.observation for step in episode.steps)
        train_act.extend(step.action for step in episode.steps)

    train_obs_v = torch.as_tensor(np.vstack(train_obs), dtype=torch.float32)
    train_act_v = torch.as_tensor(train_act, dtype=torch.long)

    return train_obs_v, train_act_v, reward_bound, reward_mean


def train_step(
    net: Net,
    optimizer: optim.Optimizer,
    objective: nn.CrossEntropyLoss,
    obs_v: torch.Tensor,
    acts_v: torch.Tensor,
) -> float:
    """One gradient update on the elite data; returns the loss value."""
    optimizer.zero_grad()
    loss_v = objective(net(obs_v), acts_v)
    loss_v.backward()
    optimizer.step()
    return loss_v.item()


def log_iteration(
    writer: SummaryWriter,
    iter_no: int,
    loss: float,
    reward_mean: float,
    reward_bound: float,
) -> None:
    print(
        f"{iter_no}: loss={loss:.3f}, "
        f"reward_mean={reward_mean:.1f} rw_bound={reward_bound:.1f}"
    )
    writer.add_scalar("loss", loss, iter_no)
    writer.add_scalar("reward_mean", reward_mean, iter_no)
    writer.add_scalar("reward_bound", reward_bound, iter_no)


def main() -> None:
    cfg = parse_args()
    # Seed before building the net: weight initialization consumes torch's RNG.
    torch.manual_seed(cfg.seed)

    # `closing` / `with` guarantee cleanup (video flush, log files) even on errors.
    with closing(make_env(cfg)) as env, SummaryWriter(comment="-cartpole") as writer:
        assert env.observation_space.shape is not None
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.spec is not None and env.spec.reward_threshold is not None
        obs_size = env.observation_space.shape[0]
        n_actions = int(env.action_space.n)
        solved_reward = env.spec.reward_threshold

        net = Net(obs_size, cfg.hidden_size, n_actions)
        print(net)
        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=net.parameters(), lr=cfg.lr)

        for iter_no, batch in enumerate(
            iterate_batches(env, net, cfg.batch_size, cfg.seed)
        ):
            obs_v, acts_v, reward_b, reward_m = filter_batch(batch, cfg.percentile)
            loss = train_step(net, optimizer, objective, obs_v, acts_v)
            log_iteration(writer, iter_no, loss, reward_m, reward_b)

            if reward_m > solved_reward:
                print("Solved!")
                break
            if cfg.max_iters is not None and iter_no + 1 >= cfg.max_iters:
                break


if __name__ == "__main__":
    main()
