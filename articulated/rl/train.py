"""Training script for RL agents.

Usage:
    # Train with raw observations (baseline)
    python -m articulated.rl.train --config articulated/configs/rl/baseline.yaml

    # Train with embeddings
    python -m articulated.rl.train --config articulated/configs/rl/embedded.yaml
"""

import argparse

import yaml

from articulated.rl.agent import RLAgent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    """Train an RL agent."""
    # Initialize agent
    agent_config = dict(config.get("agent", {}))
    if "seed" in config and "seed" not in agent_config:
        agent_config["seed"] = config["seed"]
    agent = RLAgent(**agent_config)

    # Setup
    agent.setup()

    # Train
    training_config = config.get("training", {})
    agent.train(
        total_timesteps=training_config.get("total_timesteps", 100_000),
        eval_freq=training_config.get("eval_freq", 10_000),
        save_path=training_config.get("save_path"),
    )

    # Evaluate
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    metrics = agent.evaluate(n_episodes=20)
    print(f"Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"Mean episode length: {metrics['mean_length']:.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
