"""CartPole with random actions (Gymnasium).

Headless (default) runs as fast as possible with no window.
Use --render to watch the episodes in a window.
"""
import argparse
import time

import gymnasium as gym


def run(episodes: int, render: bool, seed: int | None) -> list[float]:
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    returns = []
    for ep in range(episodes):
        # Seed only the first reset; later resets continue the same RNG stream.
        obs, info = env.reset(seed=seed if ep == 0 else None)
        if ep == 0 and seed is not None:
            env.action_space.seed(seed)
        total, steps, done = 0.0, 0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        print(f"episode {ep + 1:4d}: steps={steps:4d} return={total:6.1f}")
    env.close()
    return returns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true", help="show a window (slower)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    returns = run(args.episodes, args.render, args.seed)
    dt = time.perf_counter() - t0
    print(f"\nmean return: {sum(returns) / len(returns):.1f} over {len(returns)} episodes in {dt:.2f}s")


if __name__ == "__main__":
    main()
