"""Tests for model architectures."""

import gymnasium as gym
import pytest
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from articulated.estimation.model import GRU, LSTM, RNN, StateEstimationModel
from articulated.rl import train as rl_train
from articulated.rl.agent import RLAgent

ORIG_GYM_MAKE = gym.make


@pytest.fixture
def patch_reacher_env(monkeypatch):
    """Patch RLAgent's env creation to avoid MuJoCo in tests."""
    import articulated.rl.agent as rl_agent

    def _make_env(*args, **kwargs):
        render_mode = kwargs.get("render_mode")
        return ORIG_GYM_MAKE("Pendulum-v1", render_mode=render_mode)

    monkeypatch.setattr(rl_agent.gym, "make", _make_env)


class TestRNN:
    """Tests for the RNN architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 50
        input_size = 6
        hidden_size = 128
        output_size = 64

        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )

        x = torch.randn(batch_size, seq_length, input_size)
        output, hidden_states = model(x)

        assert output.shape == (batch_size, seq_length, output_size)
        assert hidden_states.shape == (batch_size, seq_length, hidden_size)


class TestLSTM:
    """Tests for LSTM architecture."""

    def test_not_implemented(self):
        """LSTM should raise NotImplementedError until implemented."""
        with pytest.raises(NotImplementedError):
            LSTM(input_size=6, hidden_size=64, output_size=32)


class TestGRU:
    """Tests for GRU architecture."""

    def test_not_implemented(self):
        """GRU should raise NotImplementedError until implemented."""
        with pytest.raises(NotImplementedError):
            GRU(input_size=6, hidden_size=64, output_size=32)


class TestStateEstimationModel:
    """Tests for the Lightning wrapper."""

    def test_forward_with_rnn(self):
        """Test forward pass with RNN backend."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
        )

        x = torch.randn(4, 20, 6)
        output, hidden_states = model(x)

        assert output.shape == (4, 20, 64)
        assert hidden_states.shape == (4, 20, 128)

    def test_get_embedding(self):
        """Test embedding extraction for RL."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
        )

        x = torch.randn(4, 20, 6)
        embedding = model.get_embedding(x)

        assert embedding.shape == (4, 128)

    def test_get_hidden_states(self):
        """Test hidden state extraction for analysis."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
        )

        x = torch.randn(4, 20, 6)
        hidden_states = model.get_hidden_states(x)

        assert hidden_states.shape == (4, 20, 128)


@pytest.mark.usefixtures("patch_reacher_env")
class TestRLAgent:
    """Tests for the RLAgent wrapper."""

    def test_setup_ppo(self):
        agent = RLAgent(
            algorithm="ppo",
            n_steps=32,
            batch_size=8,
            learning_rate=1e-3,
            device="cpu",
        )
        agent.setup()
        assert isinstance(agent.model, PPO)
        assert agent.env is not None
        agent.env.close()

    def test_setup_sac(self):
        agent = RLAgent(
            algorithm="sac",
            batch_size=8,
            learning_rate=1e-3,
            device="cpu",
        )
        agent.setup()
        assert isinstance(agent.model, SAC)
        assert agent.env is not None
        agent.env.close()

    def test_setup_invalid_algorithm(self):
        agent = RLAgent(algorithm="bad_algo")
        with pytest.raises(ValueError):
            agent.setup()

    def test_make_env_without_normalization(self):
        agent = RLAgent(algorithm="ppo")
        env, vec_norm = agent._make_env(render_mode=None, training=True)
        assert isinstance(env, Monitor)
        assert vec_norm is None
        env.close()

    def test_make_env_with_normalization(self):
        agent = RLAgent(algorithm="ppo", normalize_observations=True)
        env, vec_norm = agent._make_env(render_mode=None, training=True)
        assert isinstance(env, VecEnv)
        assert isinstance(vec_norm, VecNormalize)
        env.close()

    def test_sync_vec_normalize(self):
        agent = RLAgent(algorithm="ppo", normalize_observations=True)
        train_env, train_vec = agent._make_env(render_mode=None, training=True)
        agent.vec_normalize = train_vec
        eval_env, eval_vec = agent._make_env(render_mode=None, training=False)

        agent._sync_vec_normalize(eval_vec)
        assert eval_vec.obs_rms is agent.vec_normalize.obs_rms
        assert eval_vec.training is False
        assert eval_vec.norm_reward is False

        train_env.close()
        eval_env.close()

    def test_train_and_evaluate(self, tmp_path):
        agent = RLAgent(
            algorithm="ppo",
            n_steps=32,
            batch_size=8,
            learning_rate=1e-3,
            device="cpu",
            normalize_observations=True,
        )
        agent.setup()
        save_path = tmp_path / "ppo_test"
        agent.train(total_timesteps=64, eval_freq=0, save_path=str(save_path))

        assert save_path.with_suffix(".zip").exists()
        assert (tmp_path / "ppo_test_vecnormalize.pkl").exists()

        metrics = agent.evaluate(n_episodes=1)
        assert "mean_reward" in metrics
        assert "mean_length" in metrics

        agent.env.close()


class TestRLTrain:
    """Tests for the RL training entrypoint."""

    def test_train_passes_top_level_seed_to_agent(self, monkeypatch):
        captured: dict = {}

        class DummyAgent:
            def __init__(self, **kwargs):
                captured["agent_kwargs"] = kwargs

            def setup(self):
                return None

            def train(self, **kwargs):
                captured["train_kwargs"] = kwargs
                return {"total_timesteps": kwargs.get("total_timesteps", 0)}

            def evaluate(self, n_episodes=20):
                captured["n_episodes"] = n_episodes
                return {"mean_reward": 0.0, "std_reward": 0.0, "mean_length": 50.0}

        monkeypatch.setattr(rl_train, "RLAgent", DummyAgent)

        cfg = {
            "seed": 42,
            "agent": {"algorithm": "ppo"},
            "training": {"total_timesteps": 1000, "eval_freq": 100, "save_path": "x"},
        }
        rl_train.train(cfg)

        assert captured["agent_kwargs"]["seed"] == 42

    def test_train_prefers_agent_seed_over_top_level_seed(self, monkeypatch):
        captured: dict = {}

        class DummyAgent:
            def __init__(self, **kwargs):
                captured["agent_kwargs"] = kwargs

            def setup(self):
                return None

            def train(self, **kwargs):
                return {"total_timesteps": kwargs.get("total_timesteps", 0)}

            def evaluate(self, n_episodes=20):
                return {"mean_reward": 0.0, "std_reward": 0.0, "mean_length": 50.0}

        monkeypatch.setattr(rl_train, "RLAgent", DummyAgent)

        cfg = {
            "seed": 42,
            "agent": {"algorithm": "ppo", "seed": 7},
            "training": {},
        }
        rl_train.train(cfg)

        assert captured["agent_kwargs"]["seed"] == 7
